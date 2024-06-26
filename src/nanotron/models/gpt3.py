"""PyTorch GPT-3 model."""

import math
from typing import Optional
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.config import Config, GPT3Config, ParallelismArgs, Starcoder2Config
from nanotron.generation.generate_store import AttachableStore
from nanotron.models import starcoder2
from nanotron.nn.layer_norm import TritonLayerNorm
from nanotron.models.starcoder2 import MLP as Starcoder2MLP
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.models.starcoder2 import CoreAttention as Starcoder2CoreAttention
from nanotron.models.starcoder2 import GPTBlock as Starcoder2GPTBlock
from nanotron.models.starcoder2 import CausalSelfGQA, Starcoder2ForTraining, GPTModel
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
# - class GPTBlock: very similar to starcoder2 but make it so it support non-GQA or MQA
# - from starcoder import Loss


@contextmanager
def replace_coreattention(gpt3config: GPT3Config):
    orig = starcoder2.CoreAttention
    try:
        def create_core_attention(config: Starcoder2Config, parallel_config: Optional[ParallelismArgs], layer_idx: int):
            return CoreAttention(gpt3config, parallel_config, layer_idx)
        starcoder2.CoreAttention = create_core_attention
        yield
    finally:
        starcoder2.CoreAttention = orig


@contextmanager
def replace_decoder(gpt3config: GPT3Config):
    orig = starcoder2.PipelineBlock
    try:
        def create_pp_block(module_builder, module_kwargs, **kwargs):
            if module_builder is Starcoder2GPTBlock:
                # Starcoder2's GPT module is trying to instantiate a Starcoder2 GPTBlock.
                # Let's return a PipelineBlock with a GPT3Block instead.
                # This also requires to replace starcoders2's config with gpt3's config.
                module_kwargs["config"] = gpt3config
                return orig(module_builder=GPTBlock, module_kwargs=module_kwargs, **kwargs)
            # Else, they are setting up other modules, which we also want unchanged.
            return orig(module_builder=module_builder, module_kwargs=module_kwargs, **kwargs)

        starcoder2.PipelineBlock = create_pp_block
        yield
    finally:
        starcoder2.PipelineBlock = orig


@contextmanager
def replace_gpt3model(gpt3config: GPT3Config):
    orig = starcoder2.GPTModel
    try:
        def create_gptmodel(config: Starcoder2Config, parallel_context: ParallelContext,
                            parallel_config: Optional[ParallelismArgs], random_states: RandomStates):
            return GPT3Model(gpt3config, parallel_context, parallel_config, random_states)
        starcoder2.GPTModel = create_gptmodel
        yield
    finally:
        starcoder2.GPTModel = orig


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
            return attention_output.contiguous()

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
        with replace_coreattention(config):
            super().__init__(config.as_starcoder2(), parallel_config, tp_pg, layer_idx)
        self.maybe_rotary = lambda q, k, **_: (q, k)  # Overwrite possible rotary with identity.
        #self.attention = CoreAttention(config, parallel_config=parallel_config, layer_idx=layer_idx)  # Use our custom CoreAttention.


class MLP(Starcoder2MLP):
    def __init__(
        self,
        config: GPT3Config,
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
        random_states: RandomStates
    ):
        super().__init__(config.as_starcoder2(), parallel_config, tp_pg)
        self.dropout = nn.Dropout(p=config.act_pdrop)
        self.random_states = random_states
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

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
        #print("New gpt block created :D")
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
        self.ff = MLP(config=config, parallel_config=parallel_config, tp_pg=tp_pg, random_states=random_states)
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
        #hidden_states = torch.arange(hidden_states.numel()).to(hidden_states.device).to(hidden_states.dtype).view(hidden_states.size())
        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        #return {"hidden_states": hidden_states, "sequence_mask": sequence_mask}

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


class GPT3Model(GPTModel):
    def __init__(
            self,
            config: GPT3Config,
            parallel_context: ParallelContext,
            parallel_config: Optional[ParallelismArgs],
            random_states: RandomStates,
        ):
        with replace_decoder(config):
            super().__init__(config.as_starcoder2(), parallel_context, parallel_config, random_states)

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
        self.embed_scale = config.hidden_size**0.5 if config.scale_embedding else 1.0

    def forward(
        self,
        input_ids: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
        input_mask: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        input_embeds = self.token_embeddings(input_ids=input_ids, input_mask=input_mask)["input_embeds"]*self.embed_scale
        position_ids = torch.arange(input_ids.size(1), device="cuda").repeat(input_ids.size(0)).view(*input_ids.size())
        position_embeds = self.position_embeddings(position_ids=position_ids)["position_embeds"]
        hidden_states = input_embeds + position_embeds

        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.embeds_dropout(input=hidden_states)["hidden_states"]

        hidden_encoder_states = {"hidden_states": hidden_states, "sequence_mask": input_mask}
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)
        #return hidden_encoder_states["hidden_states"]

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits


# TODO: maybe reimplement:
# - tie_custom_params
# - get_embeddings_lm_head_tied_names
# - get_block_compute_costs
# - get_flops_per_sec
class GPT3ForTraining(Starcoder2ForTraining):
    def __init__(
        self,
        config: GPT3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: RandomStates,
    ):
        with replace_gpt3model(config):
            super().__init__(config.as_starcoder2(), parallel_context, parallel_config, random_states)

