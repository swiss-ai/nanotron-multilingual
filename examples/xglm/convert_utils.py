import json
from pathlib import Path
from typing import Optional

import nanotron
import torch
from nanotron.config.models_config import GPT3Config
from nanotron.models.gpt3 import GPT3ForTraining
from nanotron.trainer import mark_tied_parameters
from torch import nn


def convert_generic(module1: nn.Module, module2: nn.Module):
    names1 = {name for name, _ in module1.named_parameters()}
    names2 = {name for name, _ in module2.named_parameters()}
    assert names1 == names2, f"{names1} != {names2}"
    params2 = dict(module2.named_parameters())
    for name, param in module1.named_parameters():
        param.data = params2[name].clone()


def create_nt_model(
    model_config: Optional[GPT3Config] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[Path] = None,
):

    if model_config is None:
        assert checkpoint_path is not None
        with open(checkpoint_path / "model_config.json") as f:
            model_config = GPT3Config(**json.load(f))

    parallel_config = nanotron.config.ParallelismArgs(dp=1, pp=1, tp=1)
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    model_nt = nanotron.models.build_model(
        model_builder=lambda: GPT3ForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )
    mark_tied_parameters(model=model_nt, parallel_context=parallel_context)

    if checkpoint_path is not None:
        nanotron.serialize.load_weights(model=model_nt, parallel_context=parallel_context, root_folder=checkpoint_path)

    return model_nt
