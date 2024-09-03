import torch
import pytest

import nanotron
from nanotron.config.parallelism_config import ParallelismArgs
from nanotron.config.models_config import GPT3MoEConfig
from nanotron.parallel import ParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.trainer import mark_tied_parameters
from nanotron.models.gpt3_moe import GPT3MoEBlock, GPT3MoEForTraining
from nanotron.models.moe import LearnedRouter, dMoE

from tests.helpers.utils import init_distributed

from examples.xglm.convert_ntmoe2hf import convert_config, convert_gate, convert_ff, convert
from examples.xglm.tests.test_implementation import almost_close

from models.xglm_model import XGLMSparseMoeBlock, XGLMForCausalLM
from models.gating import BasicGate


MAX_SEQUENCE_LENGTH = 2048
TEST_SEQUENCE_LENGTH = 128  # If we test with a very large sequence length, precision errors get more significant independent of the correct implementation.
#TEST_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH
BATCH_SIZE = 4
HIDDEN_SIZE = 1024
#DTYPE = torch.bfloat16
DTYPE = torch.float32
TEXT = "Hello. This is a relatively long text. I will use this text to test the conversion scripts. Let's finish this text soon because I don't have much more to say. Final note:"

CONFIG = GPT3MoEConfig(
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
    use_spda=DTYPE is not torch.bfloat16,
    # vvv moe vvv
    is_moe=True,
    moe_num_experts=8,
    num_experts_per_tok=2,
    moe_loss_weight=0.01,
    moe_z_loss_weight=0.0,
    moe_glu=False,
)
PARALLEL_CONFIG = ParallelismArgs(dp=1, pp=1, tp=1, expert_parallel_size=1)  #CONFIG.moe_num_experts)


@pytest.fixture
def hidden_states() -> torch.Tensor:
    return torch.randn(TEST_SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE, dtype=DTYPE)


@pytest.fixture
def input_mask() -> torch.Tensor:
    return torch.ones(BATCH_SIZE, TEST_SEQUENCE_LENGTH, dtype=torch.bool)


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, CONFIG.vocab_size, (BATCH_SIZE, TEST_SEQUENCE_LENGTH))


def _test_nt2hf_gate(parallel_context: ParallelContext, hidden_states: torch.Tensor):
    hidden_states = hidden_states.cuda()

    config_hf = convert_config(CONFIG)
    gate_nt = LearnedRouter(CONFIG).cuda().to(DTYPE)
    gate_hf = BasicGate(config_hf).cuda().to(DTYPE)
    convert_gate(gate_hf, gate_nt)

    router_logits_nt, _, _ = gate_nt(hidden_states.view(-1, HIDDEN_SIZE))
    router_logits_hf = gate_hf(hidden_states.permute(1, 0, 2).reshape(-1, HIDDEN_SIZE), "")

    router_logits_nt = router_logits_nt.view(TEST_SEQUENCE_LENGTH, BATCH_SIZE, -1)
    router_logits_hf = router_logits_hf.view(BATCH_SIZE, TEST_SEQUENCE_LENGTH, -1).permute(1, 0, 2)

    assert router_logits_nt.size() == router_logits_hf.size()
    torch.testing.assert_close(router_logits_nt, router_logits_hf)


def test_nt2hf_gate(hidden_states: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt2hf_gate)(hidden_states=hidden_states)


def _test_nt2hf_ff(parallel_context: ParallelContext, hidden_states: torch.Tensor,
                   num_experts: int, num_experts_per_tok: int):
    hidden_states = hidden_states.cuda()

    config = {**vars(CONFIG)}
    config.update({"moe_num_experts": num_experts, "num_experts_per_tok": num_experts_per_tok})
    config = GPT3MoEConfig(**config)
    config_hf = convert_config(config)
    ff_nt = dMoE(config, parallel_context, PARALLEL_CONFIG).cuda().to(DTYPE)
    ff_hf = XGLMSparseMoeBlock(config_hf).cuda().to(DTYPE)
    convert_ff(ff_hf, ff_nt)

    out_nt = ff_nt(hidden_states)["hidden_states"]
    out_hf, _ = ff_hf(hidden_states.permute(1, 0, 2).contiguous(), "")
    out_hf = out_hf.permute(1, 0, 2)

    assert out_nt.size() == out_hf.size()
    almost_close(out_nt, out_hf, max_far=0.05, far_atol=0.003)


@pytest.mark.parametrize("num_experts,num_experts_per_tok", [(1, 1), (2, 1), (4, 1), (4, 2), (8, 1), (8, 2), (8, 4)])
def test_nt2hf_ff(hidden_states: torch.Tensor, num_experts: int, num_experts_per_tok: int):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt2hf_ff)(hidden_states=hidden_states, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)


def _test_nt2hf_model(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    random_states = nanotron.random.RandomStates({"tp_synced": nanotron.random.get_current_random_state()})
    input_ids = input_ids.cuda()
    input_mask = input_mask.cuda()

    # unfortunately, we can't use float64 with huggingface xglm.
    new_dtype = torch.float32 if DTYPE == torch.float64 else DTYPE

    # Get nanotron model.
    config_nt = GPT3MoEConfig(**vars(CONFIG))
    if new_dtype not in {torch.bfloat16, torch.float16}:
        config_nt.use_spda = True
    model_nt = nanotron.models.build_model(
        model_builder=lambda: GPT3MoEForTraining(
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
    model_hf = XGLMForCausalLM(convert_config(config_nt)).cuda().to(new_dtype).eval()
    convert(model_hf, model_nt)

    # Needed :/
    aux_losses = {
        "load_balancing_loss": (
            torch.zeros(1, device=input_ids.device)
            if not isinstance(input_ids, TensorPointer)
            else TensorPointer(self.input_pp_rank)
        ),
        "z_loss": (
            torch.zeros(1, device=input_ids.device)
            if not isinstance(input_ids, TensorPointer)
            else TensorPointer(self.input_pp_rank)
        ),
    }

    # Get outputs and assert.
    with torch.no_grad():
        out_nt = model_nt.model(input_ids, input_mask, aux_losses)["sharded_logits"].to(new_dtype)
        del model_nt
        torch.cuda.empty_cache()
        out_hf = model_hf(input_ids=input_ids, attention_mask=input_mask, output_router_logits=False).logits.permute(1, 0, 2)
        del model_hf
        torch.cuda.empty_cache()
    assert out_nt.size() == out_hf.size(), f"{out_nt.size()}, {out_hf.size()}"
    return out_nt.cpu(), out_hf.cpu()


def _test_nt2hf_dummy_xglm(parallel_context: ParallelContext, input_ids: torch.Tensor, input_mask: torch.Tensor):
    out_nt, out_hf = _test_nt2hf_model(parallel_context, input_ids, input_mask)
    almost_close(out_nt, out_hf, max_far=0.01, far_atol=2.0)  # We allow for less than 1% errors, but some of these are very large!
    #torch.testing.assert_close(out_nt.bfloat16(), out_hf.bfloat16())


def test_nt2hf_dummy_xglm(input_ids: torch.Tensor, input_mask: torch.Tensor):
    init_distributed(tp=1, dp=1, pp=1)(_test_nt2hf_dummy_xglm)(input_ids=input_ids, input_mask=input_mask)
