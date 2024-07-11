from typing import Optional

import nanotron
import numpy as np
import pytest
import torch
from nanotron.config.models_config import GPT3Config
from nanotron.models.gpt3 import CausalSelfAttention, GPT3ForTraining, GPTBlock, PositionEmbedding
from nanotron.parallel import ParallelContext
from nanotron.trainer import mark_tied_parameters
from transformers import XGLMTokenizer
from transformers.models.xglm.modeling_xglm import (
    XGLMAttention,
    XGLMConfig,
    XGLMDecoderLayer,
    XGLMForCausalLM,
    XGLMSinusoidalPositionalEmbedding,
)

from examples.xglm.convert_hf2nt import convert, convert_attention, convert_config, convert_decoder
from examples.xglm.convert_nt2hf import convert as convert_nt2hf
from examples.xglm.convert_nt2hf import convert_attention as convert_attention_nt2hf
from examples.xglm.convert_nt2hf import convert_config as convert_config_nt2hf
from examples.xglm.convert_nt2hf import convert_decoder as convert_decoder_nt2hf
from tests.helpers.utils import init_distributed

MAX_SEQUENCE_LENGTH = 2048
TEST_SEQUENCE_LENGTH = 128  # If we test with a very large sequence length, precision errors get more significant independent of the correct implementation.
BATCH_SIZE = 4
HIDDEN_SIZE = 1024
DTYPE = torch.float64
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
    max_position_embeddings=MAX_SEQUENCE_LENGTH,
    num_attention_heads=16,
    num_hidden_layers=24,
    scale_attn_weights=True,
    vocab_size=256008,
    sinusoidal_position_embedding=True,
    position_embedding_offset=2,
    use_spda=True,
)


@pytest.fixture
def hidden_states() -> torch.Tensor:
    return torch.randn(TEST_SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE)


@pytest.fixture
def input_mask() -> torch.Tensor:
    return torch.ones(BATCH_SIZE, TEST_SEQUENCE_LENGTH, dtype=torch.bool)


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, CONFIG.vocab_size, (BATCH_SIZE, TEST_SEQUENCE_LENGTH))


def almost_close(
    t1: torch.Tensor,
    t2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 0.016,
    max_far: float = 0.0,
    far_atol: float = 0.01,
):
    very_close = torch.abs(t1 - t2) <= atol + rtol * torch.abs(t2)
    not_very_close = ~very_close

    if torch.all(very_close):
        return
    assert (
        torch.mean(not_very_close.float()) <= max_far
    ), f"not very close found: {100*torch.mean(not_very_close.float()):.1f}%"
    assert torch.all(
        torch.abs(t1[not_very_close] - t2[not_very_close]) <= far_atol
    ), f"Worse deviation found: {torch.max(torch.abs(t1 - t2)):.4f}"


def attention_mask() -> torch.Tensor:
    # XGLM causal attention mask.
    mask = torch.ones(TEST_SEQUENCE_LENGTH, TEST_SEQUENCE_LENGTH, dtype=torch.bool, device="cuda").tril(diagonal=0)
    mask = torch.where(mask, 0.0, -np.inf).to(DTYPE)
    mask = mask.repeat(BATCH_SIZE, 1, 1).unsqueeze(1)
    return mask


##
# FROM HERE DOWN (until next comment), all tests are hf->nt
##


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
    position_ids = torch.arange(TEST_SEQUENCE_LENGTH, device="cuda").unsqueeze(0)  # shape = (1, TEST_SEQUENCE_LENGTH)

    emb_nt = PositionEmbedding(parallel_context.tp_pg, CONFIG, None).cuda()
    emb_hf = XGLMSinusoidalPositionalEmbedding(MAX_SEQUENCE_LENGTH, HIDDEN_SIZE).cuda()

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
    torch.testing.assert_close(
        out_nt.bfloat16(), out_hf.bfloat16()
    )  # We cast to bf16 to get more relaxed constraints.


def test_decoder(hidden_states: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_decoder)(hidden_states=hidden_states, sequence_mask=input_mask)


def _test_model(
    model_hf: Optional[XGLMForCausalLM],
    parallel_context: ParallelContext,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
):

    random_states = nanotron.random.RandomStates({"tp_synced": nanotron.random.get_current_random_state()})
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()

    # unfortunately, we can't use float64 with huggingface xglm.
    new_dtype = torch.float32 if DTYPE == torch.float64 else DTYPE

    # Get hf model.
    if model_hf is None:
        config_hf = XGLMConfig()
        model_hf = XGLMForCausalLM(config_hf).cuda().to(new_dtype).eval()
    else:
        model_hf = model_hf.cuda().to(new_dtype).eval()
        config_hf = model_hf.config

    # Get nanotron model and make the conversion.
    config_nt = convert_config(config_hf)
    if new_dtype not in {torch.bfloat16, torch.float16}:
        config_nt.use_spda = True
    model_nt = nanotron.models.build_model(
        model_builder=lambda: GPT3ForTraining(
            config=config_nt,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=random_states,
        ),
        parallel_context=parallel_context,
        dtype=new_dtype,
        device="cuda",
    ).eval()
    convert(model_nt, model_hf)

    print("Parameter count (M):", sum(map(torch.numel, model_hf.parameters())) / 1000 / 1000)

    # Get outputs and assert.
    with torch.no_grad():
        out_nt = model_nt.model(input_ids, input_mask).to(new_dtype)
        del model_nt
        torch.cuda.empty_cache()
        out_hf = model_hf(input_ids=input_ids, attention_mask=input_mask).logits.permute(1, 0, 2)
        del model_hf
        torch.cuda.empty_cache()
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    return out_nt.cpu(), out_hf.cpu()


def _test_dummy_xglm(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    out_nt, out_hf = _test_model(None, parallel_context, input_ids, input_mask)
    almost_close(out_nt, out_hf, max_far=0.05)


def test_dummy_xglm(input_ids: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_dummy_xglm)(input_ids=input_ids, input_mask=input_mask)


def _test_xglm500M(parallel_context: ParallelContext):
    tok = XGLMTokenizer.from_pretrained("facebook/xglm-564M")
    tokenized = tok(TEXT)
    model_hf = XGLMForCausalLM.from_pretrained("facebook/xglm-564M")
    out_nt, out_hf = _test_model(
        model_hf, parallel_context, torch.tensor([tokenized["input_ids"]]), torch.tensor([tokenized["attention_mask"]])
    )
    almost_close(out_nt, out_hf, max_far=0.1, far_atol=0.05)


def test_xglm500M():
    init_distributed(tp=1, dp=1, pp=1)(_test_xglm500M)()


def _test_xglm7B(parallel_context: ParallelContext):
    tok = XGLMTokenizer.from_pretrained("facebook/xglm-7.5B")
    tokenized = tok(TEXT)
    model_hf = XGLMForCausalLM.from_pretrained("facebook/xglm-7.5B")
    out_nt, out_hf = _test_model(
        model_hf, parallel_context, torch.tensor([tokenized["input_ids"]]), torch.tensor([tokenized["attention_mask"]])
    )
    almost_close(out_nt, out_hf, max_far=0.15, far_atol=0.1)


def test_xglm7B():
    init_distributed(tp=1, dp=1, pp=1)(_test_xglm7B)()


##
# From here down we test nt->hf converters
##


def _test_nt2hf_attention(parallel_context: ParallelContext, hidden_states: torch.Tensor, sequence_mask: torch.Tensor):
    hidden_states = hidden_states.cuda()
    sequence_mask = sequence_mask.cuda()

    attn_nt = CausalSelfAttention(CONFIG, None, parallel_context.tp_pg, 0).cuda().eval().to(DTYPE)
    attn_hf = XGLMAttention(CONFIG.hidden_size, CONFIG.num_attention_heads, CONFIG.attn_pdrop).cuda().eval().to(DTYPE)
    assert sum(map(torch.numel, attn_nt.parameters())) == sum(map(torch.numel, attn_hf.parameters()))

    convert_attention_nt2hf(attn_hf, attn_nt)
    out_nt = attn_nt(hidden_states, sequence_mask)["hidden_states"]
    out_hf = attn_hf(hidden_states.permute(1, 0, 2), attention_mask=attention_mask())[0].permute(1, 0, 2)
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    torch.testing.assert_close(out_nt, out_hf)


def test_nt2hf_attention(hidden_states: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt2hf_attention)(hidden_states=hidden_states, sequence_mask=input_mask)


def _test_nt2hf_decoder(parallel_context: ParallelContext, hidden_states: torch.Tensor, sequence_mask: torch.Tensor):
    random_states = nanotron.random.RandomStates({"tp_synced": nanotron.random.get_current_random_state()})
    hidden_states = hidden_states.cuda()
    sequence_mask = sequence_mask.cuda()

    config_hf = convert_config_nt2hf(CONFIG)
    decoder_nt = GPTBlock(CONFIG, None, parallel_context.tp_pg, random_states, 0).cuda().to(DTYPE).eval()
    decoder_hf = XGLMDecoderLayer(config_hf).cuda().to(DTYPE).eval()

    convert_decoder_nt2hf(decoder_hf, decoder_nt)

    out_nt = decoder_nt(hidden_states, sequence_mask)["hidden_states"]
    out_hf = decoder_hf(hidden_states.permute(1, 0, 2), attention_mask=attention_mask())[0].permute(1, 0, 2)

    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    torch.testing.assert_close(
        out_nt.bfloat16(), out_hf.bfloat16()
    )  # We cast to bf16 to get more relaxed constraints.


def test_nt2hf_decoder(hidden_states: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt2hf_decoder)(hidden_states=hidden_states, sequence_mask=input_mask)


def _test_nt2hf_model(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    random_states = nanotron.random.RandomStates({"tp_synced": nanotron.random.get_current_random_state()})
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()

    # unfortunately, we can't use float64 with huggingface xglm.
    new_dtype = torch.float32 if DTYPE == torch.float64 else DTYPE

    # Get nanotron model.
    config_nt = GPT3Config(**vars(CONFIG))
    if new_dtype not in {torch.bfloat16, torch.float16}:
        config_nt.use_spda = True
    model_nt = nanotron.models.build_model(
        model_builder=lambda: GPT3ForTraining(
            config=config_nt,
            parallel_context=parallel_context,
            parallel_config=None,
            random_states=random_states,
        ),
        parallel_context=parallel_context,
        dtype=new_dtype,
        device="cuda",
    ).eval()
    mark_tied_parameters(model=model_nt, parallel_context=parallel_context)

    # Create empty model_hf and make conversion.
    model_hf = XGLMForCausalLM(convert_config_nt2hf(config_nt)).cuda().to(new_dtype).eval()
    convert_nt2hf(model_hf, model_nt)

    # Get outputs and assert.
    with torch.no_grad():
        out_nt = model_nt.model(input_ids, input_mask).to(new_dtype)
        del model_nt
        torch.cuda.empty_cache()
        out_hf = model_hf(input_ids=input_ids, attention_mask=input_mask).logits.permute(1, 0, 2)
        del model_hf
        torch.cuda.empty_cache()
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    return out_nt.cpu(), out_hf.cpu()


def _test_nt2hf_dummy_xglm(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    out_nt, out_hf = _test_nt2hf_model(parallel_context, input_ids, input_mask)
    almost_close(out_nt, out_hf, max_far=0.01, far_atol=0.02)


def test_nt2hf_dummy_xglm(input_ids: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt2hf_dummy_xglm)(input_ids=input_ids, input_mask=input_mask)
