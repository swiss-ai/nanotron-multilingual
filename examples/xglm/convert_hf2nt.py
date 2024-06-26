import torch
from torch import nn

from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMConfig, XGLMDecoderLayer, XGLMForCausalLM
from nanotron.models.gpt3 import CausalSelfAttention, GPTBlock, MLP, GPT3ForTraining
from nanotron.config.models_config import GPT3Config


def convert_config(config: XGLMConfig) -> GPT3Config:
    # TODOs:
    #    dropout=0.1,
    #    layerdrop=0.0,
    #    init_std=0.02,
    #    use_cache=True,
    #    decoder_start_token_id=2,
    #    pad_token_id=1,
    #    bos_token_id=0,

    # TODO: when going gpt3->xglm:
    #  - assert layernorm is 1e-05
    return GPT3Config(
        activation_function=config.activation_function,
        attn_pdrop=config.attention_dropout,
        embd_pdrop=0.0,  # TODO
        eos_token_id=config.eos_token_id,
        hidden_size=config.d_model,
        intermediate_size=config.ffn_dim,
        layer_norm_epsilon=1e-05,
        max_position_embeddings=config.max_position_embeddings,
        num_attention_heads=config.attention_heads,
        num_hidden_layers=config.num_layers,
        resid_pdrop=0.0,  # TODO
        scale_attention_softmax_in_fp32=True,
        scale_attn_weights=True,
        vocab_size=config.vocab_size,
        sinusoidal_position_embedding=True,
        position_embedding_offset=2,
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


def convert_generic(module1: nn.Module, module2: nn.Module):
    names1 = {name for name, _ in module1.named_parameters()}
    names2 = {name for name, _ in module2.named_parameters()}
    assert names1 == names2, f"{names1} != {names2}"
    params2 = dict(module2.named_parameters())
    for name, param in module1.named_parameters():
        param.data = params2[name].clone()


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
