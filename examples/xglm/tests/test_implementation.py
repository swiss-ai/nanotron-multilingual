from typing import Optional

import numpy as np
import torch
import pytest

from transformers import XGLMTokenizer
from transformers.models.xglm.modeling_xglm import XGLMConfig, XGLMAttention, XGLMSinusoidalPositionalEmbedding, XGLMDecoderLayer, XGLMForCausalLM

import nanotron
from nanotron.config.models_config import GPT3Config
from nanotron.models.gpt3 import GPT3ForTraining, CausalSelfAttention, PositionEmbedding, GPTBlock
from nanotron.parallel import ParallelContext

from tests.helpers.utils import init_distributed

from examples.xglm.convert_hf2nt import convert_attention, convert_config, convert_decoder, convert


SEQUENCE_LENGTH = 2048
BATCH_SIZE = 4
HIDDEN_SIZE = 1024
DTYPE = torch.bfloat16
TEXT = "Hello. This is a relatively long text. I will use this text to test the conversion scripts. Let's finish this text soon because I don't have much more to say. Final note:"

CONFIG = GPT3Config(
    attn_pdrop=0.0,
    embd_pdrop=0.0,
    resid_pdrop=0.0,
    act_pdrop=0.0,
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

@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, CONFIG.vocab_size, (BATCH_SIZE, SEQUENCE_LENGTH))


def attention_mask() -> torch.Tensor:
    # XGLM causal attention mask.
    mask = torch.ones(SEQUENCE_LENGTH, SEQUENCE_LENGTH, dtype=torch.bool, device="cuda").tril(diagonal=0)
    mask = torch.where(mask, 0.0, -np.inf).to(DTYPE)
    mask = mask.repeat(BATCH_SIZE, 1, 1).unsqueeze(1)
    return mask


def _test_attention(parallel_context: ParallelContext, hidden_states: torch.Tensor, sequence_mask: torch.Tensor):
    hidden_states = hidden_states.cuda()
    sequence_mask = sequence_mask.cuda()

    attn_nt = CausalSelfAttention(CONFIG, None, parallel_context.tp_pg, 0).cuda().eval().to(DTYPE)
    attn_hf = XGLMAttention(CONFIG.hidden_size, CONFIG.num_attention_heads, CONFIG.attn_pdrop).cuda().eval().to(DTYPE)
    assert sum(map(torch.numel, attn_nt.parameters())) == sum(map(torch.numel, attn_hf.parameters()))

    convert_attention(attn_nt, attn_hf)
    out_nt = attn_nt(hidden_states, sequence_mask)["hidden_states"]
    out_hf = attn_hf(hidden_states.permute(1, 0, 2), attention_mask=attention_mask())[0].permute(1, 0, 2)
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


def _test_decoder(parallel_context: ParallelContext, hidden_states: torch.Tensor, sequence_mask: torch.Tensor):
    random_states = nanotron.random.RandomStates({"tp_synced": nanotron.random.get_current_random_state()})
    hidden_states = hidden_states.cuda()
    sequence_mask = sequence_mask.cuda()

    config_hf = XGLMConfig()
    decoder_hf = XGLMDecoderLayer(config_hf).cuda().to(DTYPE).eval()
    config_nt = convert_config(config_hf)
    if DTYPE not in {torch.bfloat16, torch.float16}:
        config_nt.use_spda = True
    decoder_nt = GPTBlock(config_nt, None, parallel_context.tp_pg, random_states, 0).cuda().to(DTYPE).eval()

    convert_decoder(decoder_nt, decoder_hf)

    out_nt = decoder_nt(hidden_states, sequence_mask)["hidden_states"]
    out_hf = decoder_hf(hidden_states.permute(1, 0, 2), attention_mask=attention_mask())[0].permute(1, 0, 2)

    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    torch.testing.assert_close(out_nt, out_hf)


def test_decoder(hidden_states: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_decoder)(hidden_states=hidden_states, sequence_mask=input_mask)


def _test_model(model_hf: Optional[XGLMForCausalLM], parallel_context: ParallelContext,
                input_ids: torch.Tensor, input_mask: torch.Tensor):
    random_states = nanotron.random.RandomStates({"tp_synced": nanotron.random.get_current_random_state()})
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()

    # Get hf model.
    if model_hf is None:
        config_hf = XGLMConfig()
        model_hf = XGLMForCausalLM(config_hf).cuda().to(DTYPE).eval()
    else:
        model_hf = model_hf.cuda().to(DTYPE).eval()
        config_hf = model_hf.config

    # Get nanotron model and make the conversion.
    config_nt = convert_config(config_hf)
    if DTYPE not in {torch.bfloat16, torch.float16}:
        config_nt.use_spda = True
    model_nt = nanotron.models.build_model(
        model_builder=lambda: GPT3ForTraining(
            config=config_nt,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=random_states,
        ),
        parallel_context=parallel_context,
        dtype=DTYPE,
        device="cuda",
    ).eval()
    convert(model_nt, model_hf)

    print("Parameter count (M):", sum(map(torch.numel, model_hf.parameters()))/1000/1000)

    # Get outputs and assert.
    with torch.no_grad():
        out_nt = model_nt.model(input_ids, input_mask).to(DTYPE)
        del model_nt
        torch.cuda.empty_cache()
        out_hf = model_hf(input_ids=input_ids, attention_mask=input_mask).logits.permute(1, 0, 2)
        del model_hf
        torch.cuda.empty_cache()
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    torch.testing.assert_close(out_nt.cpu(), out_hf.cpu())

def _test_dummy_xglm(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    _test_model(None, parallel_context, input_ids, input_mask)


def test_dummy_xglm(input_ids: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_dummy_xglm)(input_ids=input_ids, input_mask=input_mask)


def _test_xglm7B(parallel_context: ParallelContext):
    tok = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
    tokenized = tok(TEXT)
    model_hf = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B")
    _test_model(model_hf, parallel_context,
                torch.tensor([tokenized["input_ids"]]), torch.tensor([tokenized["attention_mask"]]))


def test_xglm7B():
    init_distributed(tp=1, dp=1, pp=1)(_test_xglm7B)()


def _test_xglm500M(parallel_context: ParallelContext):
    tok = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
    tokenized = tok(TEXT)
    model_hf = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
    _test_model(model_hf, parallel_context,
                torch.tensor([tokenized["input_ids"]]), torch.tensor([tokenized["attention_mask"]]))


def test_xglm500M():
    init_distributed(tp=1, dp=1, pp=1)(_test_xglm500M)()
