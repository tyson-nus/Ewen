from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class StructuredLinear(nn.Module):


    def __init__(self, base: nn.Linear, rank: int = 16):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Structured adapters require stored nn.Linear matrices")
        if rank < 1:
            raise ValueError("Residual rank must be positive")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        row_norms = base.weight.detach().float().norm(dim=1)
        active = torch.nonzero(row_norms > 0, as_tuple=False).flatten()
        self.register_buffer("row_norms", row_norms)
        self.register_buffer("active_rows", active)
        self.c = nn.Parameter(torch.zeros(active.numel(), device=base.weight.device, dtype=torch.float32))
        self.A = nn.Parameter(torch.empty(rank, self.in_features, device=base.weight.device,
                                          dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank, device=base.weight.device,
                                          dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):

        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def full_coefficients(self) -> torch.Tensor:
        return self.c.new_zeros(self.out_features).index_copy(0, self.active_rows, self.c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        weight_output = base_output if self.base.bias is None else base_output - self.base.bias
        scales = self.full_coefficients() / self.row_norms.clamp_min(torch.finfo(torch.float32).tiny)
        core_output = weight_output * scales.to(weight_output.dtype)
        residual = F.linear(F.linear(inputs.to(self.A.dtype), self.A), self.B).to(base_output.dtype)
        return base_output + core_output + residual


class ResidualLinear(nn.Module):


    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                          dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                          dtype=torch.float32))
        self.scale = float(alpha) / rank
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        base_output = self.base(inputs)
        residual = F.linear(F.linear(inputs.to(self.A.dtype), self.A), self.B).to(base_output.dtype)
        return base_output + self.scale * residual


FULL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LINEAR_TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


def text_config(backbone):
    config = backbone.config
    return getattr(config, "text_config", config)


def decoder_layers(backbone):

    candidates = []
    for name, module in backbone.named_modules():
        if isinstance(module, nn.ModuleList) and name.split(".")[-1] == "layers" and len(module):
            if all(hasattr(layer, "mlp") and (hasattr(layer, "self_attn") or
                                             hasattr(layer, "linear_attn")) for layer in module):
                candidates.append((name, module))
    if len(candidates) != 1:
        raise ValueError(f"Expected one text-decoder layer list, found {len(candidates)}")
    return candidates[0]


@dataclass(frozen=True)
class AdapterReport:
    mode: str
    target_names: tuple[str, ...]
    core_count: int
    residual_count: int


def attach_structured_adapters(backbone: nn.Module, rank: int = 16,
                               mode: str = "structured", lora_alpha: float = 32.0) -> AdapterReport:

    if mode not in {"structured", "lora", "full", "frozen"}:
        raise ValueError(f"Unknown adaptation mode: {mode}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(mode == "full")
    if mode in {"full", "frozen"}:
        return AdapterReport(mode, (), 0, 0)
    prefix, layers = decoder_layers(backbone)
    layer_types = getattr(text_config(backbone), "layer_types", None)
    if layer_types is None or len(layer_types) != len(layers):
        raise ValueError("The backbone must declare one layer_type per text decoder layer")
    target_names = []
    for index, (layer, layer_type) in enumerate(zip(layers, layer_types)):
        if layer_type == "full_attention":
            attention_name, names = "self_attn", FULL_TARGETS
        elif layer_type == "linear_attention":
            attention_name, names = "linear_attn", LINEAR_TARGETS
        else:
            raise ValueError(f"Unsupported decoder layer type: {layer_type}")
        for component_name, matrix_names in ((attention_name, names), ("mlp", MLP_TARGETS)):
            component = getattr(layer, component_name, None)
            if component is None:
                raise ValueError(f"Decoder layer {index} lacks configured component {component_name}")
            for matrix_name in matrix_names:
                base = getattr(component, matrix_name, None)
                if not isinstance(base, nn.Linear):
                    raise TypeError(f"Configured matrix {component_name}.{matrix_name} is not nn.Linear")
                adapter = (StructuredLinear(base, rank) if mode == "structured" else
                           ResidualLinear(base, rank, lora_alpha))
                setattr(component, matrix_name, adapter)
                target_names.append(f"{prefix}.{index}.{component_name}.{matrix_name}")
    cores = core_parameters(backbone)
    residuals = residual_parameters(backbone)
    return AdapterReport(mode, tuple(target_names), sum(p.numel() for p in cores),
                         sum(p.numel() for p in residuals))


def core_parameters(model: nn.Module) -> list[nn.Parameter]:

    return [module.c for module in model.modules() if isinstance(module, StructuredLinear)
            and module.c.numel()]


def residual_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for module in model.modules()
            if isinstance(module, (StructuredLinear, ResidualLinear))
            for parameter in (module.A, module.B)]


def parameter_groups(model: nn.Module, learning_rate: float, weight_decay: float = .1):
    cores, residuals = core_parameters(model), residual_parameters(model)
    claimed = {id(p) for p in cores + residuals}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in claimed]
    return [{"params": cores, "lr": learning_rate, "weight_decay": weight_decay, "name": "core"},
            {"params": residuals, "lr": learning_rate, "weight_decay": weight_decay, "name": "residual"},
            {"params": other, "lr": learning_rate, "weight_decay": weight_decay, "name": "eeg"}]
