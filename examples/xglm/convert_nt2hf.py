"""
Converts a nanotron model to HF format
Command:
    torchrun --nproc-per-node=1 convert_nt2hf.py --checkpoint-path=nanotron_weights --save-path=hf_weights
"""

from argparse import ArgumentParser
from typing import Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMConfig, XGLMDecoderLayer, XGLMForCausalLM

from nanotron.config.models_config import GPT3Config
from nanotron.models.gpt3 import CausalSelfAttention, GPTBlock, MLP, GPT3ForTraining
from examples.xglm.convert_utils import convert_generic, create_nt_model


def convert_config(config: GPT3Config) -> XGLMConfig:
    if config.embd_pdrop != config.resid_pdrop:
        warnings.warn(f"nanotron.embd_pdrop = {config.embd_pdrop} does not match with "
                      f"nanotron.resid_pdrop = {config.resid_pdrop}. "
                      "XGLM implementation needs these two values to be equal "
                      "for correct conversion.")
    if config.layer_norm_epsilon != 1e-5:
        warnings.warn(f"nanotron.layer_norm_epsilon must be 1e-5, not {config.layer_norm_epsilon}")
    return XGLMConfig(
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
    )


def convert_attention(attn_hf: XGLMAttention, attn_nt: XGLMAttention):
    qs_w = []
    ks_w = []
    vs_w = []
    qs_b = []
    ks_b = []
    vs_b = []

    head_dim = attn_hf.head_dim
    qkv_ws = list(attn_nt.query_key_value.weight.split(head_dim))
    qkv_bs = list(attn_nt.query_key_value.bias.split(head_dim))
    for i, (w, b) in enumerate(zip(qkv_ws, qkv_bs)):
        if i % 3 == 0:
            qs_w.append(w)
            qs_b.append(b)
        elif i % 3 == 1:
            ks_w.append(w)
            ks_b.append(b)
        else:
            vs_w.append(w)
            vs_b.append(b)

    q_w = torch.cat(qs_w)
    k_w = torch.cat(ks_w)
    v_w = torch.cat(vs_w)
    q_b = torch.cat(qs_b)
    k_b = torch.cat(ks_b)
    v_b = torch.cat(vs_b)
    
    with torch.no_grad():
        attn_hf.q_proj.weight.data = q_w.clone()
        attn_hf.k_proj.weight.data = k_w.clone()
        attn_hf.v_proj.weight.data = v_w.clone()
        attn_hf.q_proj.bias.data = q_b.clone()
        attn_hf.k_proj.bias.data = k_b.clone()
        attn_hf.v_proj.bias.data = v_b.clone()

        attn_hf.out_proj.weight.data = attn_nt.dense.weight.data.clone()
        attn_hf.out_proj.bias.data = attn_nt.dense.bias.data.clone()


def convert_decoder(block_hf: XGLMDecoderLayer, block_nt: GPTBlock):
    convert_generic(block_hf.self_attn_layer_norm, block_nt.ln_1)
    convert_attention(block_hf.self_attn, block_nt.attn)
    convert_generic(block_hf.final_layer_norm, block_nt.ln_2)
    convert_generic(block_hf.fc1, block_nt.ff.c_fc)
    convert_generic(block_hf.fc2, block_nt.ff.c_proj)


def convert(model_hf: XGLMForCausalLM, model_nt: GPT3ForTraining):
    convert_generic(model_hf.model.embed_tokens, model_nt.model.token_embeddings.pp_block.token_embedding)
    for layer_hf, layer_nt in zip(model_hf.model.layers, model_nt.model.decoder):
        convert_decoder(layer_hf, layer_nt.pp_block)
    convert_generic(model_hf.model.layer_norm, model_nt.model.final_layer_norm.pp_block)
    convert_generic(model_hf.lm_head, model_nt.model.lm_head.pp_block)


def main(checkpoint_path: Path, save_path: Path, tokenizer_name: Optional[str]):
    # Load nanotron model.
    model_nt = create_nt_model(checkpoint_path=checkpoint_path)

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
    parser.add_argument("--checkpoint-path", type=Path, default="checkpoints/xglm-7.5B", help="Path to the nanotron checkpoint")
    parser.add_argument("--save-path", type=Path, default="facebook/xglm-7.5B", help="Path to save the huggingface model")
    parser.add_argument("--tokenizer-name", type=str, default="facebook/xglm-7.5B")
    args = parser.parse_args()
    main(args.checkpoint_path, args.save_path, args.tokenizer_name)

