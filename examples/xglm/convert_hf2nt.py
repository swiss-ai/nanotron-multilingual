import torch

from transformers.models.xglm.modeling_xglm import XGLMAttention
from nanotron.models.gpt3 import CausalSelfAttention


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
