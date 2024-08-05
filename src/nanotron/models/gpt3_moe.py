"""PyTorch GPT-3 MoE model."""

import math
from contextlib import contextmanager
from typing import Dict, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from nanotron import distributed as dist
from nanotron.config import GPT3MoEConfig, ParallelismArgs, GPT3Config
from nanotron.generation.generate_store import AttachableStore
from nanotron.models import gpt3
from nanotron.models.moe import (
    dMoE,
)
from nanotron.models.gpt3 import CausalSelfAttention, GPTModel, PositionEmbedding, dropout_add_fused_train, GPT3ForTraining
from nanotron.models.gpt3 import GPTBlock as GPT3Block
from nanotron.nn.layer_norm import TritonLayerNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.block import PipelineBlock
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelColumnLinear, TensorParallelEmbedding
from nanotron.random import RandomStates, branch_random_state


@contextmanager
def replace_decoder(gpt3config: GPT3MoEConfig):
    orig = gpt3.PipelineBlock
    try:

        def create_pp_block(module_builder, module_kwargs, **kwargs):
            if module_builder is GPT3Block:
                # GPT3's GPT module is trying to instantiate a GPT3 GPTBlock.
                # Let's return a PipelineBlock with a GPT3Block instead.
                # This also requires to replace starcoders2's config with gpt3's config.
                module_kwargs["config"] = gpt3config
                return orig(module_builder=GPTBlock, module_kwargs=module_kwargs, **kwargs)
            # Else, they are setting up other modules, which we also want unchanged.
            return orig(module_builder=module_builder, module_kwargs=module_kwargs, **kwargs)

        gpt3.PipelineBlock = create_pp_block
        yield
    finally:
        gpt3.PipelineBlock = orig


@contextmanager
def replace_gpt3model(gpt3moeconfig: GPT3MoEConfig):
    orig = gpt3.GPTModel
    try:

        def create_gptmodel(
            config: GPT3Config,
            parallel_context: ParallelContext,
            parallel_config: Optional[ParallelismArgs],
            random_states: RandomStates,
        ):
            return GPT3MoEModel(gpt3moeconfig, parallel_context, parallel_config, random_states)

        gpt3.GPTModel = create_gptmodel
        yield
    finally:
        gpt3.GPTModel = orig

class GPTBlock(nn.Module):
    def __init__(
        self,
        config: GPT3MoEConfig,
        parallel_config: Optional[ParallelismArgs],
        parallel_context: ParallelContext,
        tp_pg: dist.ProcessGroup,
        random_states: RandomStates,
        layer_idx: int,
    ):
        super(GPTBlock, self).__init__()
        self.ln_1 = TritonLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(
            config=config, parallel_config=parallel_config, tp_pg=tp_pg, layer_idx=layer_idx
        )
        self.attn_dropout = config.attn_pdrop

        self.ln_2 = TritonLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.ff = dMoE(
                config=config,
                parallel_config=parallel_config,
                parallel_context=parallel_context,
        )
        self.ff_dropout = config.resid_pdrop
        self.random_states = random_states
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        
    def forward(
        self,
        hidden_states: torch.Tensor | TensorPointer,
        sequence_mask: torch.Tensor | TensorPointer,
        aux_losses: Dict[str, Union[torch.Tensor, TensorPointer]],
    ) -> dict[str, torch.Tensor | TensorPointer]:

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # hidden_states = torch.arange(hidden_states.numel()).to(hidden_states.device).to(hidden_states.dtype).view(hidden_states.size())
        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        # return {"hidden_states": hidden_states, "sequence_mask": sequence_mask}

        if self.training:
            with branch_random_state(
                self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
            ):
                hidden_states = dropout_add_fused_train(hidden_states, residual=residual, prob=self.attn_dropout)
        else:
            # No need for random state context manager
            hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.ff(hidden_states=hidden_states)
        hidden_states = mlp_output["hidden_states"]

        for key, value in mlp_output.items():
            if key != "hidden_states":
                aux_losses[key] = aux_losses[key] + value

        if self.training:
            with branch_random_state(
                self.random_states, "tp_synced", enabled=self.tp_mode is TensorParallelLinearMode.ALL_REDUCE
            ):
                hidden_states = dropout_add_fused_train(hidden_states, residual=residual, prob=self.ff_dropout)
        else:
            # No need for random state context manager
            hidden_states = hidden_states + residual

        return {
            "hidden_states": hidden_states,
            "sequence_mask": output["sequence_mask"],
            "aux_losses": aux_losses
        }

class GPT3MoEModel(GPTModel):
    def __init__(
        self,
        config: GPT3MoEConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: RandomStates,
    ):
        with replace_decoder(config):
            super().__init__(config.as_gpt3(), parallel_context, parallel_config, random_states)
        
        # need to adapt the decoder list because we pass the aux_losses around
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
                        "parallel_context": parallel_context,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "sequence_mask", "aux_losses"},
                    module_output_keys={"hidden_states", "sequence_mask", "aux_losses"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
    def forward(
        self,
        input_ids: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
        input_mask: torch.Tensor | TensorPointer,  # [batch_size, seq_length]
        aux_losses: Dict[str, Union[torch.Tensor, TensorPointer]],
    ):
        # all tensors are optional as most ranks don't need anything from the dataloader.

        input_embeds = (
            self.token_embeddings(input_ids=input_ids, input_mask=input_mask)["input_embeds"] * self.embed_scale
        )
        # TODO: position_ids could be cached.
        position_ids = torch.arange(input_ids.size(1), device="cuda").repeat(input_ids.size(0)).view(*input_ids.size())
        position_embeds = self.position_embeddings(position_ids=position_ids)["position_embeds"]
        hidden_states = input_embeds + position_embeds

        with branch_random_state(
            self.random_states, "tp_synced", enabled=self.tp_mode == TensorParallelLinearMode.ALL_REDUCE
        ):
            hidden_states = self.embeds_dropout(input=hidden_states)["hidden_states"]

        hidden_encoder_states = {"hidden_states": hidden_states, "sequence_mask": input_mask, "aux_losses": aux_losses}
        for encoder_block in self.decoder:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)
        # return hidden_encoder_states["hidden_states"]

        hidden_states = self.final_layer_norm(input=hidden_encoder_states["hidden_states"])["hidden_states"]

        sharded_logits = self.lm_head(x=hidden_states)["logits"]

        fp32_sharded_logits = self.cast_to_fp32(x=sharded_logits)["output"]

        return fp32_sharded_logits, hidden_encoder_states["aux_losses"]


# TODO: maybe reimplement:
# - get_block_compute_costs
# - get_flops_per_sec
class GPT3MoEForTraining(GPT3ForTraining):
    def __init__(
        self,
        config: GPT3MoEConfig,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: RandomStates,
    ):
        with replace_gpt3model(config):
            super().__init__(config.as_gpt3(), parallel_context, parallel_config, random_states)
        self.config = config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        input_mask: Union[torch.Tensor, TensorPointer],
        label_ids: Union[torch.Tensor, TensorPointer],
        label_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        # aux_losses are used for load balancing in case of MoEs
        aux_losses = {
            "load_balancing_loss": (
                torch.zeros(1, device=input_ids.device)
                if not isinstance(input_ids, TensorPointer)
                else TensorPointer(self.input_pp_rank)
            ),
            "z_loss": (
                torch.zeros(1, device=input_ids.device)
                if not isinstance(input_ids, TensorPointer)
                else TensorPointer(self.input_pp_rank)
            ),
        }
        output = self.model(
            input_ids=input_ids,
            input_mask=input_mask,
            aux_losses=aux_losses,
        )
        loss = self.loss(
            sharded_logits=output["sharded_logits"],
            label_ids=label_ids,
            label_mask=label_mask,
        )
        
        if isinstance(output['aux_losses'], dict):
            for key, value in output["aux_losses"].items():
                loss[key] = value
        return loss
    
    # TODO: adapt with MoE costs
    def get_block_compute_costs(self):
        """Computes the compute cost of each block in the model so that we can do a better job of load balancing."""
        model_config = self.config
        d_ff = model_config.n_inner if model_config.intermediate_size is not None else 4 * model_config.hidden_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        block_compute_costs = {
            # CausalSelfAttention (qkv proj + attn out) + MLP
            GPTBlock: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 2 * d_ff * model_config.hidden_size,
            # This is the last lm_head
            TensorParallelColumnLinear: model_config.vocab_size * model_config.hidden_size,
        }
        return block_compute_costs