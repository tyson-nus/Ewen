from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class ProjectionDiagnostics:
    reference_count: int
    ridge: float
    reference_energy: float
    task_norm: float
    projected_norm: float
    solve_dtype: str
    angular_deviation_before: list[float]
    angular_deviation_after: list[float]


def angular_deviation(update: torch.Tensor, reference: torch.Tensor) -> float:
    denom = update.double().norm() * reference.double().norm()
    if float(denom) == 0:
        return 0.0
    ratio = (torch.dot(update.double(), reference.double()).abs() / denom).clamp(0, 1)
    return float(torch.asin(ratio))


def attenuate_vector(task: torch.Tensor, references: torch.Tensor, rho: float = .5):

    if task.ndim != 1 or references.ndim != 2 or references.shape[0] != task.numel():
        raise ValueError("Expected task[D] and reference[D,K]")
    if references.shape[1] < 1 or rho <= 0 or not math.isfinite(rho):
        raise ValueError("A nonempty reference and positive ridge coefficient are required")
    if not torch.isfinite(task).all() or not torch.isfinite(references).all():
        raise FloatingPointError("Nonfinite core task/reference gradient; optimizer step aborted")
    device = task.device
    reference = references.to(device=device, dtype=torch.float32)
    gradient = task.float()
    gram = reference.T @ reference
    energy = gram.trace() / reference.shape[1]
    ridge = float(rho * energy)
    solve_dtype = "float32"
    if not bool(torch.count_nonzero(references)):
        projected = gradient.clone()
    else:
        def solve(dtype):
            matrix = references.to(device=device, dtype=dtype)
            vector = task.to(dtype=dtype)
            hessian = matrix.T @ matrix
            alpha = rho * hessian.trace() / matrix.shape[1]
            regularized = hessian + alpha * torch.eye(matrix.shape[1], device=device, dtype=dtype)
            factor, info = torch.linalg.cholesky_ex(regularized)
            if int(info.max()) != 0 or not torch.isfinite(factor).all():
                raise FloatingPointError("Reference Gram Cholesky factorization failed")
            weights = torch.cholesky_solve((matrix.T @ vector).unsqueeze(1), factor).squeeze(1)
            result = vector - matrix @ weights
            if not torch.isfinite(result).all():
                raise FloatingPointError("Reference attenuation produced nonfinite gradient")
            return result
        try:
            if float(energy) <= 0 or not math.isfinite(float(energy)):
                raise FloatingPointError("Nonzero reference Gram energy underflowed or overflowed")
            projected = solve(torch.float32)
        except (RuntimeError, FloatingPointError):
            solve_dtype = "float64"
            projected = solve(torch.float64).float()
            precise_reference = references.to(device=device, dtype=torch.float64)
            energy = (precise_reference.T @ precise_reference).trace() / references.shape[1]
            ridge = float(rho * energy)
    diagnostics = ProjectionDiagnostics(
        reference_count=reference.shape[1], ridge=ridge, reference_energy=float(energy),
        task_norm=float(gradient.norm()), projected_norm=float(projected.norm()),
        solve_dtype=solve_dtype,
        angular_deviation_before=[angular_deviation(gradient, col) for col in reference.T],
        angular_deviation_after=[angular_deviation(projected, col) for col in reference.T],
    )
    return projected.to(task.dtype), diagnostics


def flatten_gradients(params: Sequence[nn.Parameter], gradients=None) -> torch.Tensor:
    if not params:
        raise ValueError("Core parameter list is empty")
    gradients = [p.grad for p in params] if gradients is None else list(gradients)
    if len(params) != len(gradients):
        raise ValueError("Gradient and parameter list lengths differ")
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).float()
                      for p, g in zip(params, gradients)])


def reference_columns(params: Sequence[nn.Parameter], reference_grads) -> torch.Tensor:
    if not reference_grads:
        raise ValueError("At least one text reference microbatch is required")
    return torch.stack([flatten_gradients(params, gradient) for gradient in reference_grads], dim=1)


def project_core_gradients(params: Sequence[nn.Parameter], reference_grads,
                           rho: float = .5, expected_k: int = 4):

    if len(reference_grads) != expected_k:
        raise ValueError(f"Expected {expected_k} distinct OWT microbatches, got {len(reference_grads)}")
    task = flatten_gradients(params)
    references = reference_columns(params, reference_grads)
    projected, diagnostics = attenuate_vector(task, references, rho)
    position = 0
    for parameter in params:
        size = parameter.numel()
        parameter.grad = projected[position:position + size].view_as(parameter).to(parameter.dtype)
        position += size
    return diagnostics
