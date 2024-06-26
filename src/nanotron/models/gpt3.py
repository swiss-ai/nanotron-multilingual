"""PyTorch GPT-3 model."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.config import Config, GPT3Config, ParallelismArgs
from nanotron.generation.generate_store import AttachableStore
from nanotron.models.starcoder2 import MLP as Starcoder2MLP
from nanotron.models.starcoder2 import CoreAttention as Starcoder2CoreAttention
from nanotron.models.starcoder2 import CausalSelfGQA
from nanotron.random import RandomStates, branch_random_state
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelEmbedding
from nanotron.parallel.tied_parameters import tie_parameters

# NOTES:
# - tie head_weight with embeddings I think.

# TODO:
# - class GPT3Config: config lol
# - check that attention (i.e. nanotron.attn vs xglm.self_attn) is the same.
# - from starcoder import Embedding
# - class PositionEmbedding: my sinusoidal embedding extends from TensorParallelEmbedding
# - class GPTBLock: very similar to starcoder2 but make it so it support non-GQA or MQA
# - from starcoder import Loss


class CoreAttention(Starcoder2CoreAttention):
    def __init__(self, config: GPT3Config, parallel_config: Optional[ParallelismArgs], layer_idx: int):
        super().__init__(config.as_starcoder2(), parallel_config, layer_idx)
        self.gpt3config = config

    def forward(self, 
        query_states: torch.Tensor,  # [batch_size * q_length, q_heads, inner_dim]
        key_states: torch.Tensor,  # [batch_size * kv_length, kv_heads, inner_dim]
        value_states: torch.Tensor,  # [batch_size * kv_length, kv_heads, inner_dim]
        q_sequence_mask: torch.Tensor,  # torch.BoolTensor [batch_size, q_length] (can be broadcasted to that size)
        kv_sequence_mask: torch.Tensor,  # torch.BoolTensor [batch_size, kv_length] (can be broadcasted to that size)
    ):

        if self.gpt3config.use_spda:
            assert torch.all(q_sequence_mask)
            assert torch.all(kv_sequence_mask)

            batch_size, q_length = q_sequence_mask.size()
            kv_length = kv_sequence_mask.size(1)
            _, q_heads, head_dim = query_states.size()
            kv_heads = key_states.size(1)

            attention_output = F.scaled_dot_product_attention(
                query_states.view(batch_size, q_length, q_heads, head_dim).permute(0, 2, 1, 3),
                key_states.view(batch_size, kv_length, kv_heads, head_dim).permute(0, 2, 1, 3),
                value_states.view(batch_size, kv_length, kv_heads, head_dim).permute(0, 2, 1, 3),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )  # [batch, q_length, q_heads, head_dim]
            attention_output = attention_output.permute(0, 2, 1, 3)
            attention_output = attention_output.reshape(batch_size*q_length, q_heads, head_dim)
            return attention_output

        assert query_states.dtype in {torch.bfloat16, torch.float16}
        return super().forward(query_states, key_states, value_states, q_sequence_mask, kv_sequence_mask)


class CausalSelfAttention(CausalSelfGQA):
    def __init__(
        self,
        config: GPT3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        layer_idx: int,
    ):
        super().__init__(config.as_starcoder2(), parallel_config, tp_pg, layer_idx)
        self.maybe_rotary = lambda q, k, **_: (q, k)  # Overwrite possible rotary with identity.
        self.attention = CoreAttention(config, parallel_config=parallel_config, layer_idx=layer_idx)  # Use our custom CoreAttention.


class MLP(Starcoder2MLP):
    def __init__(
        self,
        config: GPT3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        # TODO: GPT3Config -> Starcoder2Config.
        super().__init__(config, parallel_config, tp_pg)
        self.dropout = nn.Dropout(p=config.dropout) # TODO: correct config.dropout name

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.dropout(input=hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return {"hidden_states": hidden_states}


class GPTBlock(nn.Module):
    def __init__(
        self,
        config: GPT3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        random_states: RandomStates,
        layer_idx: int,
    ):
        super(GPTBlock, self).__init__()
        self.ln_1 = TritonLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx
        )
        self.attn_dropout = config.attn_pdrop

        self.ln_2 = TritonLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ff = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg)
        self.ff_dropout = config.resid_pdrop

        self.random_states = random_states
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

    def forward(
        self,
        hidden_states: torch.Tensor | TensorPointer,
        sequence_mask: torch.Tensor | TensorPointer,
    ) -> dict[str, torch.Tensor | TensorPointer]:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]

        if self.training:
            with branch_random_state(
                self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
            ):
                hidden_states = dropout_add_fused_train(hidden_states, residual=residual, prob=self.attn_dropout)
        else:
            # No need for random state context manager
            # TODO: add dropout scaling?
            hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.ff(hidden_states=hidden_states)["hidden_states"]

        if self.training:
            with branch_random_state(
                self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
            ):
                hidden_states = dropout_add_fused_train(hidden_states, residual=residual, prob=self.ff_dropout)
        else:
            # No need for random state context manager
            # TODO: add dropout scaling?
            hidden_states = hidden_states + residual

        return {
            "hidden_states": hidden_states,
            "sequence_mask": output["sequence_mask"],
        }


class PositionEmbedding(nn.Module, AttachableStore):
    def __init__(self, tp_pg: dist.ProcessGroup, config: GPT3Config, parallel_config: Optional[ParallelismArgs]):
        super().__init__()

        self.config = config
        if (config.max_position_embeddings + config.position_embedding_offset) % tp_pg.size() == 0:
            dummy_pos = 0
        else:
            dummy_pos = tp_pg.size() - ((config.max_position_embeddings + config.position_embedding_offset) % k)
        true_max_size = config.max_position_embeddings + config.position_embedding_offset + dummy_pos

        if config.sinusoidal_position_embedding:
            weight = self._make_weights(tp_pg, true_max_size, config.hidden_size)
        else:
            weight = None

        position_embedding = TensorParallelEmbedding(
            num_embeddings=true_max_size,
            embedding_dim=config.hidden_size,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
            _weight=weight
        )
        self.pg = tp_pg

        # Sinusoidal position embeddings are usually not trainable.
        # We adjust that by setting the module self.position_embedding without gradient.
        if config.sinusoidal_position_embedding:
            with torch.no_grad():
                self.position_embedding = position_embedding.requires_grad_(False)
        else:
            self.position_embedding = position_embedding

    def forward(self, position_ids: torch.Tensor):  # [batch_size, seq_length]
        position_ids = position_ids.transpose(0, 1)
        position_embeds = self.position_embedding(position_ids + self.config.position_embedding_offset)
        return {"position_embeds": position_embeds}

    def _make_weights(self, tp_pg: dist.ProcessGroup, num_embeddings: int,
                      embedding_dim: int) -> torch.Tensor:
        rank = dist.get_rank(group=tp_pg)
        tp_size = tp_pg.size()

        assert 0 <= rank < tp_size
        assert num_embeddings % tp_size == 0
        assert embedding_dim % 2 == 0
        block_size = num_embeddings//tp_size

        half_dim = embedding_dim//2
        emb = math.log(10_000)/(half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = (rank*block_size + torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1)) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(block_size, embedding_dim)
        return emb


class GPT3Model(nn.Module):
    def __init__(
            self,
            config: GPT3Config,
            parallel_context: ParallelContext,
            parallel_config: Optional[ParallelismArgs],
            random_states: RandomStates,
        ):
        super().__init__()

        # Declare all the nodes
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.random_states = random_states
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        self.token_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=Embedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"input_ids"},
            module_output_keys={"input_embeds"},
        )
        self.position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=PositionEmbedding,
            module_kwargs={
                "tp_pg": parallel_context.tp_pg,
                "config": config,
                "parallel_config": parallel_config,
            },
            module_input_keys={"position_ids"},
            module_output_keys={"position_embeds"},
        )

        self.embeds_dropout = PipelineBlock(
            p2p=self.p2p,
            module_builder=nn.Dropout,
            module_kwargs={"p": config.embd_pdrop},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=GPTBlock,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "random_states": random_states,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "sequence_mask"},
                    module_output_keys={"hidden_states", "sequence_mask"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonLayerNorm,
            module_kwargs={"normalized_shape": config.hidden_size, "eps": config.layer_norm_epsilon},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.lm_head = PipelineBlock(
            p2p=self.p2p,
            # Understand that this means that we return sharded logits that are going to need to be gathered
            module_builder=TensorParallelColumnLinear,
            module_kwargs={
                "in_features": config.hidden_size,
                "out_features": config.vocab_size,
                "pg": parallel_context.tp_pg,
                "bias": False,
                # TODO: refactor so that we store that default in a single place.
                "mode": self.tp_mode,
                "async_communication": parallel_config.tp_linear_async_communication
                if parallel_config is not None
                else False,
            },
            module_input_keys={"x"},
            module_output_keys={"logits"},
        )

        self.cast_to_fp32 = PipelineBlock(
            p2p=self.p2p,
            module_builder=lambda: lambda x: x.float(),
            module_kwargs={},
            module_input_keys={"x"},
            module_output_keys={"output"},
        )


    def forward(
        self,
        input_ids: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
        input_mask: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        position_ids = torch.arange(input_ids.size(1), device="cuda").repeat(input_ids.size(0)).view(*input_ids.size())
        input_embeds = self.token_embeddings(input_ids=input_ids)["input_embeds"]
        position_embeds = self.position_embeds(position_ids=position_ids)["position_embeds"]
        hidden_states = input_embeds + position_embeds

        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.embeds_dropout(input=hidden_states)["hidden_states"]

        hidden_encoder_states = {"hidden_states": hidden_states, "sequence_mask": input_mask}
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits
