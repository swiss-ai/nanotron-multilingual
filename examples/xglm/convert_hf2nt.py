"""
Converts a HF model to nanotron format
Command:
    torchrun --nproc-per-node=1 convert_hf2nt.py --checkpoint-path=hf_weights --save-path=nanotron_weights
"""

import json
import warnings
import dataclasses
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMConfig, XGLMDecoderLayer, XGLMForCausalLM

import nanotron
from nanotron.models.gpt3 import CausalSelfAttention, GPTBlock, MLP, GPT3ForTraining
from nanotron.config.models_config import GPT3Config
from nanotron.trainer import mark_tied_parameters
from examples.xglm.convert_utils import convert_generic, create_nt_model


def convert_config(config: XGLMConfig) -> GPT3Config:
    # These settings seem to be unused:
    #    layerdrop=0.0,
    #    init_std=0.02,
    #    use_cache=True,
    #    pad_token_id=1,
    #    bos_token_id=0,
    if config.dropout != config.attention_dropout:
        warnings.warn(f"huggingface.dropout = {config.dropout} does not match with "
                      f"huggingface.attention_dropout = {config.attention_dropout}. "
                      "Nanotron implementation needs these two values to be equal "
                      "for correct conversion.")
    return GPT3Config(
        activation_function=config.activation_function,
        attn_pdrop=config.attention_dropout,
        embd_pdrop=config.dropout,
        eos_token_id=config.eos_token_id,
        hidden_size=config.d_model,
        intermediate_size=config.ffn_dim,
        layer_norm_epsilon=1e-05,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.attention_heads,
        num_hidden_layers=config.num_layers,
        resid_pdrop=config.dropout,
        scale_attention_softmax_in_fp32=True,
        scale_attn_weights=True,
        vocab_size=config.vocab_size,
        sinusoidal_position_embedding=True,
        position_embedding_offset=config.decoder_start_token_id,
        use_spda=False,
        act_pdrop=config.activation_dropout,
        scale_embedding=config.scale_embedding,
    )


def convert_attention(attn_nt: CausalSelfAttention, attn_hf: XGLMAttention):
    q_ws = torch.chunk(attn_hf.q_proj.weight, attn_hf.num_heads)
    k_ws = torch.chunk(attn_hf.k_proj.weight, attn_hf.num_heads)
    v_ws = torch.chunk(attn_hf.v_proj.weight, attn_hf.num_heads)

    q_bs = torch.chunk(attn_hf.q_proj.bias, attn_hf.num_heads)
    k_bs = torch.chunk(attn_hf.k_proj.bias, attn_hf.num_heads)
    v_bs = torch.chunk(attn_hf.v_proj.bias, attn_hf.num_heads)

    qkv_w = []
    qkv_b = []
    for q_w, k_w, v_w, q_b, k_b, v_b in zip(q_ws, k_ws, v_ws, q_bs, k_bs, v_bs):
        qkv_w += [q_w, k_w, v_w]
        qkv_b += [q_b, k_b, v_b]
    qkv_w = torch.cat(qkv_w)
    qkv_b = torch.cat(qkv_b)

    with torch.no_grad():
        attn_nt.query_key_value.weight.data = qkv_w.clone()
        attn_nt.query_key_value.bias.data = qkv_b.clone()
        attn_nt.dense.weight.data = attn_hf.out_proj.weight.clone()
        attn_nt.dense.bias.data = attn_hf.out_proj.bias.clone()


def convert_mlp(mlp_nt: MLP, block_hf: XGLMDecoderLayer):
    convert_generic(mlp_nt.c_fc, block_hf.fc1)
    convert_generic(mlp_nt.c_proj, block_hf.fc2)


def convert_decoder(block_nt: GPTBlock, block_hf: XGLMDecoderLayer):
    convert_generic(block_nt.ln_1, block_hf.self_attn_layer_norm)
    convert_attention(block_nt.attn, block_hf.self_attn)
    convert_generic(block_nt.ln_2, block_hf.final_layer_norm)
    convert_mlp(block_nt.ff, block_hf)


def convert(model_nt: GPT3ForTraining, model_hf: XGLMForCausalLM):
    convert_generic(model_nt.model.token_embeddings.pp_block.token_embedding, model_hf.model.embed_tokens)
    for layer_nt, layer_hf in zip(model_nt.model.decoder, model_hf.model.layers):
        convert_decoder(layer_nt.pp_block, layer_hf)
    convert_generic(model_nt.model.final_layer_norm.pp_block, model_hf.model.layer_norm)
    convert_generic(model_nt.model.lm_head.pp_block, model_hf.lm_head)


def main(hf_path: str, save_path: Path):
    # Load hf.
    print("Loading hf...")
    model_hf = XGLMForCausalLM.from_pretrained(hf_path)

    # Init nanotron.
    print("Initializing nt...")
    config_nt = convert_config(model_hf.config)
    model_nt = create_nt_model(config_nt)

    # Copy weights and save model.
    print("Copying weights...")
    convert(model_nt, model_hf)
    nanotron.serialize.save_weights(model=model_nt, parallel_context=model_nt.parallel_context,
                                    root_folder=save_path)
    with open(save_path/"model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(config_nt), f)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert HF weights to nanotron format")
    parser.add_argument("--checkpoint-path", default="facebook/xglm-7.5B", help="Name or path to the huggingface checkpoint")
    parser.add_argument("--save-path", type=Path, default="checkpoints/xglm-7.5B", help="Path to save the nanotron model")
    args = parser.parse_args()
    main(args.checkpoint_path, args.save_path)
