import numpy as np
import torch
import pytest

from transformers.models.xglm.modeling_xglm import XGLMAttention, XGLMSinusoidalPositionalEmbedding

from nanotron.config.models_config import GPT3Config
from nanotron.models.gpt3 import CausalSelfAttention, PositionEmbedding
from nanotron.parallel import ParallelContext

from tests.helpers.utils import init_distributed

from examples.xglm.convert_hf2nt import convert_attention


SEQUENCE_LENGTH = 2048
BATCH_SIZE = 4
HIDDEN_SIZE = 1024
DTYPE = torch.float64

CONFIG = GPT3Config(
    attn_pdrop=0.0,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    eos_token_id=2,
    hidden_size=HIDDEN_SIZE,
    intermediate_size=4096,
    layer_norm_epsilon=1e-05,
    max_position_embeddings=SEQUENCE_LENGTH,
    num_attention_heads=16,
    num_hidden_layers=24,
    scale_attn_weights=True,
    vocab_size=256008,
    sinusoidal_position_embedding=True,
    position_embedding_offset=2,
    use_spda=True
)


@pytest.fixture
def hidden_states() -> torch.Tensor:
    return torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE,
                       dtype=DTYPE)


@pytest.fixture
def input_mask() -> torch.Tensor:
    return torch.ones(BATCH_SIZE, SEQUENCE_LENGTH, dtype=torch.bool)


def _test_attention(parallel_context: ParallelContext, hidden_states: torch.Tensor, sequence_mask: torch.Tensor):
    hidden_states = hidden_states.cuda()
    sequence_mask = sequence_mask.cuda()

    attn_nt = CausalSelfAttention(CONFIG, None, parallel_context.tp_pg, 0).cuda().eval().to(DTYPE)
    attn_hf = XGLMAttention(CONFIG.hidden_size, CONFIG.num_attention_heads, CONFIG.attn_pdrop).cuda().eval().to(DTYPE)
    assert sum(map(torch.numel, attn_nt.parameters())) == sum(map(torch.numel, attn_hf.parameters()))

    # Build xglm mask.
    mask = torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool, device="cuda").tril(diagonal=0)
    mask = torch.where(mask, 0.0, -np.inf).to(DTYPE)
    mask = mask.repeat(BATCH_SIZE, 1, 1).unsqueeze(1)

    convert_attention(attn_nt, attn_hf)
    out_nt = attn_nt(hidden_states, sequence_mask)["hidden_states"]
    out_hf = attn_hf(hidden_states.permute(1, 0, 2), attention_mask=mask)[0].permute(1, 0, 2)
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    torch.testing.assert_close(out_nt, out_hf)


def test_attention(hidden_states: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_attention)(hidden_states=hidden_states, sequence_mask=input_mask)


def _test_position_embeddings(parallel_context: ParallelContext):
    position_ids = torch.arange(SEQUENCE_LENGTH, device="cuda").unsqueeze(0)  # shape = (1, SEQUENCE_LENGTH)

    emb_nt = PositionEmbedding(parallel_context.tp_pg, CONFIG, None).cuda()
    emb_hf = XGLMSinusoidalPositionalEmbedding(SEQUENCE_LENGTH, HIDDEN_SIZE).cuda()

    assert emb_nt.position_embedding.weight.size() == emb_hf.weights.size()
    torch.testing.assert_close(emb_nt.position_embedding.weight, emb_hf.weights)

    out_nt = emb_nt(position_ids)["position_embeds"]
    out_hf = emb_hf(position_ids).permute(1, 0, 2)
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    torch.testing.assert_close(out_nt, out_hf)

def test_position_embeddings():
    init_distributed(tp=1, dp=1, pp=1)(_test_position_embeddings)()
