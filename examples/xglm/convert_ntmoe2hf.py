"""
Converts a nanotron moe model to HF format
Command:
    torchrun --nproc-per-node=1 convert_nt2hf.py --checkpoint-path=nanotron_weights --save-path=hf_weights
"""

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from nanotron.config.models_config import GPT3MoEConfig
from nanotron.models.gpt3_moe import GPT3MoEForTraining, GPT3MoEBlock
from nanotron.models.moe import dMoE, SparseMLP

from examples.xglm.convert_dense2moe import create_nt_moe_model, convert_attention
from examples.xglm.convert_utils import convert_generic

from models.xglm_model import XGLMForCausalLM, XGLMDecoderLayer, XGLMmoeConfig, XGLMSparseMoeBlock, XGLMMLP

# TODO: nanotron moe scales down the moe weights but hf doesn't
# TODO: nanotron does not use pdrop in moe.


def convert_config(config: GPT3MoEConfig) -> XGLMmoeConfig
    assert config.moe_num_experts > 1, f"Why are you using a 1-expert moe? lol"
    if config.embd_pdrop != config.resid_pdrop:
        warnings.warn(
            f"nanotron.embd_pdrop = {config.embd_pdrop} does not match with "
            f"nanotron.resid_pdrop = {config.resid_pdrop}. "
            "XGLM implementation needs these two values to be equal "
            "for correct conversion."
        )
    if config.layer_norm_epsilon != 1e-5:
        warnings.warn(f"nanotron.layer_norm_epsilon must be 1e-5, not {config.layer_norm_epsilon}")
    if config.moe_z_loss_weight != 0:
        warnings.warn(f"transformer implementation does not support z loss")
    assert not config.moe_glu, "Transformer implementation does not support glu MLP layers"

    return XGLMmoeConfig(
        # Regular xglm config.
        activation_function=config.activation_function,
        attention_dropout=config.attn_pdrop,
        dropout=config.embd_pdrop,
        eos_token_id=config.eos_token_id,
        d_model=config.hidden_size,
        ffn_dim=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        attention_heads=config.num_attention_heads,
        num_layers=config.num_hidden_layers,
        vocab_size=config.vocab_size,
        decoder_start_token_id=config.position_embedding_offset,
        activation_dropout=config.act_pdrop,
        scale_embedding=config.scale_embedding,
        # Moe specifics.
        num_local_experts=config.moe_num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        gate_type="linear",
        gate_depth=1,
        router_aux_loss_coef=config.moe_looss_weight,
    )


def convert_mlp(mlp_hf: XGLMMLP, mlp_nt: SparseMLP):
    # TODO: mlp_hf has non-zero bias.
    convert_generic(mlp_hf.fc1, mlp_nt.w1.module)
    convert_generic(mlp_hf.fc2, mlp_nt.w2.module)


def convert_ff(ff_hf: XGLMSparseMoeBlock, ff_nt: dMoE):
    convert_generic(ff_hf.gate.gate, ff_nt.router.layer)
    for expert_hf, expert_nt in zip(ff_hf.experts, ff_nt.experts):
        convert_mlp(expert_hf, expert_nt.mlp)


def convert_decoder(block_hf: XGLMDecoderLayer, block_nt: GPT3MoEBlock):
    convert_generic(block_hf.self_attn_layer_norm, block_nt.ln_1)
    convert_attention(block_hf.self_attn, block_nt.attn)
    convert_generic(block_hf.final_layer_norm, block_nt.ln_2)
    # TODO: hf has fc1, fc2 attributes but they are not used, probably should be removed.
    convert_generic(block_hf.fc1, block_nt.ff.c_fc)
    convert_generic(block_hf.fc2, block_nt.ff.c_proj)


def convert(model_hf: XGLMForCausalLM, model_nt: GPT3MoEForTraining):
    convert_generic(model_hf.model.embed_tokens, model_nt.model.token_embeddings.pp_block.token_embedding)
    for layer_hf, layer_nt in zip(model_hf.model.layers, model_nt.model.decoder):
        convert_decoder(layer_hf, layer_nt.pp_block)
    convert_generic(model_hf.model.layer_norm, model_nt.model.final_layer_norm.pp_block)
    convert_generic(model_hf.lm_head, model_nt.model.lm_head.pp_block)


def main(checkpoint_path: Path, save_path: Path, tokenizer_name: Optional[str]):
    # Load nanotron model.
    model_nt = create_nt_moe_model(checkpoint_path=checkpoint_path)

    # Init huggingface model.
    model_config_hf = convert_config(model_nt.config)
    model_hf = XGLMForCausalLM._from_config(model_config_hf)

    # Copy weights, initialize tokenizer and save model.
    if tokenizer_name is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(save_path)
    convert(model_hf, model_nt)
    model_hf.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert HF weights to nanotron format")
    parser.add_argument(
        "--checkpoint-path", type=Path, default="checkpoints/xglm-7.5B", help="Path to the nanotron checkpoint"
    )
    parser.add_argument(
        "--save-path", type=Path, default="facebook/xglm-7.5B", help="Path to save the huggingface model"
    )
    parser.add_argument("--tokenizer-name", type=str, default="facebook/xglm-7.5B")
    args = parser.parse_args()
    main(args.checkpoint_path, args.save_path, args.tokenizer_name)
