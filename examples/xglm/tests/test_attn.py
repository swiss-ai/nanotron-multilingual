import torch
from torch.nn import functional as F
#torch.Size([4, 2048, 16, 64]), torch.Size([2048, 4, 1024])

# inputs = (batchsize * qlen, heads, head_dim)
# outputs = (batchsize*qlen, heads, head_dim)
def sdpa(query, key, value, batchsize: int):
    def reshape(tensor):  # output = (batchsize, heads, qlen, head_dim)
        return tensor.view(batchsize, qlen, heads, head_dim).permute(0, 2, 1, 3)

    batchsize_x_qlen, heads, head_dim = query.size()
    qlen = batchsize_x_qlen//batchsize
    out = F.scaled_dot_product_attention(reshape(query), reshape(key), reshape(value), is_causal=True)  # (b,h,q,d)
    return out.permute(0, 2, 1, 3).reshape(batchsize*qlen, heads, head_dim)


# inputs = (batchsize * qlen, heads, head_dim)
# outputs = (batchsize*qlen, heads, head_dim)
def fa(query_states, key_states, value_states, batchsize: int):
    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    batchsize_x_qlen, heads, head_dim = query_states.size()
    qlen = batchsize_x_qlen//batchsize

    q_sequence_mask = torch.ones(batchsize, qlen, dtype=torch.bool, device="cuda")
    kv_sequence_mask = torch.ones(batchsize, qlen, dtype=torch.bool, device="cuda")

    # TODO @thomasw21: Compute once, instead of computing for each layers.
    cu_seqlens_q = torch.zeros((q_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
    cu_seqlens_k = torch.zeros((kv_sequence_mask.shape[0] + 1), dtype=torch.int32, device=query_states.device)
    torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_q[1:])
    torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32, out=cu_seqlens_k[1:])

    # TODO(kunhao): flash attn's causal means that the query can only attend to the keys before it. This is not
    # what we want if we are using kv cache. This is a hack as we always have q_length == 1 when using kv cache.
    causal = False if q_sequence_mask.shape[1] == 1 else True
    attn_output = flash_attn_varlen_func(
        q=query_states,
        k=key_states,
        v=value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=q_sequence_mask.shape[1],
        max_seqlen_k=kv_sequence_mask.shape[1],
        dropout_p=0.0,
        softmax_scale=None,  # defaults to 1/sqrt(d_qk)
        causal=causal,
        window_size=(-1, -1),
        return_attn_probs=False,
    )
    return attn_output


def main():
    batchsize = 5
    qlen = 6
    heads = 2
    head_dim = 16

    query = torch.randn(batchsize*qlen, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(batchsize*qlen, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(batchsize*qlen, heads, head_dim, device="cuda", dtype=torch.bfloat16)

    out_pt = sdpa(query, key, value, batchsize)
    out_fa = fa(query, key, value, batchsize)

    assert out_pt.size() == out_fa.size()

    torch.testing.assert_close(out_pt, out_fa)



if __name__ == "__main__":
    main()
