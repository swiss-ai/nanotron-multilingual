"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc-per-node=1 convert_dense2moe.py --checkpoint-path=nanotron_weights --save-path=nanotron_moe_weights
"""

import dataclasses
import json
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from torch import nn
import torch
import nanotron
from nanotron.config.models_config import GPT3Config, GPT3LangMoEConfig
from nanotron.models.gpt3 import GPT3ForTraining, GPTBlock
from nanotron.models.gpt3_langmoe import GPT3LangMoEForTraining, GPT3LangMoEBlock
from nanotron.trainer import mark_tied_parameters

from convert_utils import convert_generic, create_nt_model


def convert_config(config: GPT3Config, num_experts=8, num_languages=32, language_embedding_size=128) -> GPT3LangMoEConfig:
    return GPT3LangMoEConfig(
        **config.__dict__,
        is_moe=True,
        moe_num_experts=num_experts,
        num_experts_per_tok=min(2, num_experts),  # arbitrarily chosen
        moe_loss_weight=0.01,  # arbitrarily chosen
        moe_z_loss_weight=0.001,  # arbitrarily chosen
        moe_glu=False,
        num_languages=num_languages,
        language_embedding_size=language_embedding_size,
    )


def convert_dense_to_moe(ff_moe: nn.Module, dense_ff: nn.Module, num_experts: int):
    with torch.no_grad():
        # only copy the weight matrix and repeat it n_expert times
        weight_1 = dense_ff.c_fc.weight.clone()
        if num_experts == 1:
            ff_moe.experts.mlp.w1.module.weight.data = weight_1.contiguous()
        else:
            # [intermediate_size, hidden_size] -> [hidden_size, intermediate_size * n_experts]
            weight_1 = weight_1.T
            ff_moe.experts.mlp.w1.module.weight.data = weight_1.repeat(1, num_experts)

        weight_2 = dense_ff.c_proj.weight.clone()
        if num_experts == 1: # just a specific case for 1 expert
            ff_moe.experts.mlp.w2.module.weight.data = weight_2.contiguous()
        else: 
            # [hidden_size, intermediate_size] -> [intermediate_size * n_experts, hidden_size]
            weight_2 = weight_2.T
            ff_moe.experts.mlp.w2.module.weight.data = weight_2.repeat(num_experts, 1)

        # # -- could add bias only for 2nd layer, because that works with the MegaBlocks MoE implementation
        # # -- but won't make a big difference?
        # ff_moe.experts.bias.copy_(dense_ff.c_proj.bias)

        # init gating randomly
        nn.init.normal_(ff_moe.gate.layer.weight, mean=0.0, std=0.02)


def convert_decoder(block_moe: GPT3LangMoEBlock, block_nt: GPTBlock, num_experts: int):
    convert_generic(block_moe.ln_1, block_nt.ln_1)
    convert_generic(block_moe.attn, block_nt.attn)
    convert_generic(block_moe.ln_2, block_nt.ln_2)
    convert_dense_to_moe(block_moe.ff, block_nt.ff, num_experts)


def convert(
    model_moe: GPT3LangMoEForTraining, model_dense: GPT3ForTraining, num_experts: int
):
    convert_generic(
        model_moe.model.token_embeddings.pp_block.token_embedding,
        model_dense.model.token_embeddings.pp_block.token_embedding,
    )
    # init laguage embedding randomly
    nn.init.normal_(model_moe.model.language_embeddings.pp_block.language_embedding.weight, mean=0.0, std=0.02)
    for layer_moe, layer_nt in zip(model_moe.model.decoder, model_dense.model.decoder):
        convert_decoder(layer_moe.pp_block, layer_nt.pp_block, num_experts)
    convert_generic(
        model_moe.model.final_layer_norm.pp_block,
        model_dense.model.final_layer_norm.pp_block,
    )
    convert_generic(
        model_moe.model.lm_head.pp_block, model_dense.model.lm_head.pp_block
    )


def create_nt_moe_model(
    model_config: Optional[GPT3Config] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[Path] = None,
):

    if model_config is None:
        assert checkpoint_path is not None
        with open(checkpoint_path / "model_config.json") as f:
            model_config = GPT3LangMoEConfig(**json.load(f))

    parallel_config = nanotron.config.ParallelismArgs(dp=1, pp=1, tp=1)
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    model_nt = nanotron.models.build_model(
        model_builder=lambda: GPT3LangMoEForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )
    mark_tied_parameters(model=model_nt, parallel_context=parallel_context)

    if checkpoint_path is not None:
        nanotron.serialize.load_weights(
            model=model_nt,
            parallel_context=parallel_context,
            root_folder=checkpoint_path,
        )

    return model_nt


def main(
    checkpoint_path: Path,
    save_path: Path,
    num_experts: int,
    num_languages: int,
    language_embedding_size: int,
):
    # Load nanotron model.
    model_dense = create_nt_model(checkpoint_path=checkpoint_path)

    # Init moe model.
    model_config_moe = convert_config(model_dense.config, num_experts, num_languages, language_embedding_size)
    model_moe = create_nt_moe_model(model_config=model_config_moe)

    convert(model_moe, model_dense, num_experts)
    nanotron.serialize.save_weights(
        model=model_moe,
        parallel_context=model_moe.parallel_context,
        root_folder=save_path,
    )
    with open(save_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(model_config_moe), f)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # fix all random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    parser = ArgumentParser(description="Convert dense weights to moe format")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default="checkpoints/xglm-7.5B",
        help="Path to the nanotron dense checkpoint",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default="checkpoints/xglm-moe-7.5B",
        help="Path to save the nanotron moe model",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts in the MoE model (duplicates of MLP layer)",
    )
    parser.add_argument(
        "--num-languages",
        type=int,
        default=32,
        help="Number of languages for the language embedding",
    )
    parser.add_argument(
        "--language-embedding-size",
        type=int,
        default=128,
        help="Size of the language embedding",
    )
    args = parser.parse_args()
    main(args.checkpoint_path, args.save_path, args.num_experts, args.num_languages, args.language_embedding_size)
