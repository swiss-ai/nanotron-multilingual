import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from abc import ABC, abstractmethod


class Gate(ABC):
    def __init__(self, device):
        super(Gate, self).__init__()
        self.device = device

    @abstractmethod
    def compute(self, x):
        """
        Compute the output of the gate.
        This method should be implemented by all subclasses.
        """
        pass


def init_x_embeddings(Xs, x_embedding_dim):
    x2embeddings = nn.ParameterDict(dict())
    for x in Xs:
        x_embedding = torch.empty(x_embedding_dim)
        nn.init.normal_(x_embedding)
        x2embeddings[str(x)] = nn.Parameter(x_embedding)
    return x2embeddings


class BasicGate(nn.Module):
    """One or two layer feedforward network as the Gate."""

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.ffn_dim = config.ffn_dim
        self.activation = nn.ReLU(self.ffn_dim)

        if config.gate_depth == 1:
            self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        elif config.gate_depth == 2:
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.ffn_dim),
                self.activation,
                nn.Linear(self.ffn_dim, self.num_experts, bias=False),
            )
        else:
            raise ValueError("Invalid gate_depth!")

    def forward(self, x, lang_name):
        return self.gate(x)


class LanguageAwareGate(nn.Module):
    """One or two layer feedforward network as the Gate."""

    def __init__(self, config) -> None:
        super().__init__()

        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.ffn_dim = config.ffn_dim
        self.activation = nn.ReLU(self.ffn_dim)
        self.language_embedding_dim = (
            config.language_embedding_dim
            if config.language_embedding_dim is not None
            else config.hidden_size
        )
        self.lang_embeddings = init_x_embeddings(
            config.languages, self.language_embedding_dim
        )

        if config.gate_depth == 1:
            self.gate = nn.Linear(
                self.hidden_dim + self.language_embedding_dim,
                self.num_experts,
                bias=False,
            )
        elif config.gate_depth == 2:
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.ffn_dim),
                self.activation,
                nn.Linear(self.ffn_dim, self.num_experts, bias=False),
            )
        else:
            raise ValueError("Invalid gate_depth!")

    def forward(self, x, lang_name):
        # TODO x needs to be added to the language embedding (we need to pass the language as well)
        lang_embedding = self.lang_embeddings[str(lang_name)]
        lang_embedding.squeeze(0)
        lang_embedding = lang_embedding.expand(x.shape[0], -1)
        x = torch.cat((x, lang_embedding), dim=-1)
        return self.gate(x)


class TopKGate(Gate):
    def __init__(self, device, straight_through, k=1):
        super(TopKGate, self).__init__(device)
        self.k = k
        self.device = device
        self.straight_through = straight_through

    def compute(self, x):
        if self.k > 1:
            topk_gate_scores, indices = torch.topk(x, self.k)
            topk_gate_scores = F.softmax(
                topk_gate_scores,
                dim=1,
                dtype=torch.float,
            ).type_as(x)
            mask = F.one_hot(indices, x.shape[-1]).float()
            mask_flat = mask.sum(dim=-1)
            combine_tensor = (
                topk_gate_scores[..., None, None, None]
                * mask_flat[..., None, None, None]
                * F.one_hot(indices, x.shape[-1])[..., None, None]
            )
            combine_tensor = combine_tensor.sum(1)
            return combine_tensor, indices, topk_gate_scores
        elif self.k == 1:
            x = F.softmax(x, dim=-1)
            topk_gate_scores, index = x.topk(
                k=self.k, dim=-1
            )  # torch.nn.functional.softmax(x , dim=-1).topk(k=self.k, dim=-1)
            if self.straight_through:
                index_soft = F.softmax(x, dim=-1)
                index = (index - index_soft).detach() + index_soft
                index = index[:, 0]
                topk_gate_scores, index = map(
                    lambda x: x.squeeze(dim=-1), (topk_gate_scores, index)
                )
            else:
                topk_gate_scores, index = map(
                    lambda x: x.squeeze(dim=-1), (topk_gate_scores, index)
                )

            mask = F.one_hot(index, x.shape[-1]).float()
            mask_flat = mask.sum(dim=-1)
            combine_tensor = (
                topk_gate_scores[..., None, None, None]
                * mask_flat[..., None, None, None]
                * F.one_hot(index, x.shape[-1])[..., None, None]
            )
            return combine_tensor, index, topk_gate_scores
