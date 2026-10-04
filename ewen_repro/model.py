from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import attach_structured_adapters, core_parameters, residual_parameters, text_config


@dataclass
class EwenOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    cov_loss: Optional[torch.Tensor] = None
    generation_loss: Optional[torch.Tensor] = None
    generation_loss_sum: Optional[torch.Tensor] = None
    generation_token_count: int = 0
    valid_target_tokens: int = 0
    valid_covariate_windows: int = 0
    hidden_states: Optional[torch.Tensor] = None


def masked_pool(states: torch.Tensor, mask: torch.Tensor):
    weights = mask.to(states.dtype).unsqueeze(-1)
    return (states * weights).sum(1) / weights.sum(1).clamp_min(1)


class CovariateEncoder(nn.Module):


    def __init__(self, width: int, bottleneck: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(25, bottleneck), nn.GELU(), nn.Linear(bottleneck, width))
        self.modulation = nn.Sequential(nn.Linear(2 * width, bottleneck), nn.GELU(),
                                        nn.Linear(bottleneck, 2 * width))

        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, descriptors, availability, descriptor_mask=None):
        if descriptors.ndim != 2 or descriptors.shape[-1] != 22:
            raise ValueError("Descriptors must be [batch,22], with EEG/EOG/ECG/EMG blocks")
        if availability.shape != (descriptors.shape[0], 3):
            raise ValueError("Auxiliary availability must be [batch,3]")
        if not torch.all((availability == 0) | (availability == 1)):
            raise ValueError("Availability indicators must remain binary")
        valid = torch.isfinite(descriptors)
        if descriptor_mask is not None:
            if descriptor_mask.shape != descriptors.shape:
                raise ValueError("Per-value descriptor mask must match [batch,22]")
            valid = valid & descriptor_mask.bool()
        valid = valid.clone()
        for index, (start, stop) in enumerate(((7, 12), (12, 17), (17, 22))):
            valid[:, start:stop] &= availability[:, index:index + 1].bool()
        clean = torch.where(valid, descriptors, torch.zeros_like(descriptors)).float()
        return self.encoder(torch.cat((clean, availability.float()), dim=-1)), valid

    def apply_modulation(self, states, covariate):
        expanded = covariate.unsqueeze(1).expand(-1, states.shape[1], -1)
        gamma, beta = self.modulation(torch.cat((expanded, states), dim=-1)).chunk(2, dim=-1)
        return (1 + gamma) * states + beta


class TokenGeometry(nn.Module):
    def __init__(self, width: int, heads: int, max_times: int = 64, max_channels: int = 256):
        super().__init__()
        self.max_times, self.max_channels, self.heads = max_times, max_channels, heads
        position = torch.arange(max_times, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000) / width))
        encoding = torch.zeros(max_times, width)
        encoding[:, 0::2] = torch.sin(position * scales)
        encoding[:, 1::2] = torch.cos(position * scales[:encoding[:, 1::2].shape[1]])
        self.register_buffer("time_encoding", encoding)
        self.channel_encoding = nn.Embedding(max_channels, width)
        bottleneck = min(128, width)
        self.global_projection = nn.Sequential(nn.Linear(width, bottleneck, bias=False), nn.GELU(),
                                               nn.Linear(bottleneck, width, bias=False))
        self.temporal_bias = nn.Embedding(2 * max_times - 1, heads)
        self.channel_bias = nn.Embedding(max_channels * max_channels, heads)
        nn.init.zeros_(self.temporal_bias.weight)
        nn.init.zeros_(self.channel_bias.weight)

    def validate(self, times, channels, mask):
        if times.shape != channels.shape or times.shape != mask.shape:
            raise ValueError("Time/channel IDs and EEG valid mask must have the same shape")
        if mask.any() and ((times[mask] < 0).any() or (times[mask] >= self.max_times).any()):
            raise ValueError("EEG time IDs exceed geometry capacity")
        if mask.any() and ((channels[mask] < 0).any() or (channels[mask] >= self.max_channels).any()):
            raise ValueError("EEG channel IDs exceed geometry capacity")

    def forward(self, times, channels, covariate, mask, enabled=True):
        self.validate(times, channels, mask)
        safe_times = times.masked_fill(~mask, 0)
        safe_channels = channels.masked_fill(~mask, 0)
        if not enabled:
            batch, tokens = times.shape
            position = covariate.new_zeros(batch, tokens, covariate.shape[-1])
            bias = covariate.new_zeros(batch, self.heads, tokens, tokens)
            return position, bias
        position = (self.time_encoding[safe_times] + self.channel_encoding(safe_channels)
                    + self.global_projection(covariate).unsqueeze(1))
        distance = (safe_times.unsqueeze(2) - safe_times.unsqueeze(1)).clamp(
            -(self.max_times - 1), self.max_times - 1) + self.max_times - 1
        pair = safe_channels.unsqueeze(2) * self.max_channels + safe_channels.unsqueeze(1)
        bias = (self.temporal_bias(distance) + self.channel_bias(pair)).permute(0, 3, 1, 2)
        return position, bias


class EEGAttention(nn.Module):


    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("EEG interface width must divide attention head count")
        inner = min(128, width)
        inner = max(heads, inner // heads * heads)
        self.width, self.inner, self.heads = width, inner, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * inner)
        self.output = nn.Linear(inner, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, inner), nn.GELU(),
                                nn.Linear(inner, width))

    def forward(self, states, mask, bias):
        batch, tokens, width = states.shape
        qkv = self.qkv(self.norm(states)).reshape(batch, tokens, 3, self.heads, self.inner // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        additive = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=additive,
                                                  dropout_p=0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.inner)
        output = states + self.output(attended)
        output = output + self.ff(output)
        return output.masked_fill(~mask.unsqueeze(-1), 0)


class TemporalClassifier(nn.Module):
    def __init__(self, width: int, classes: int, depth: int = 2, hidden: int = 128):
        super().__init__()
        if depth not in {2, 4}:
            raise ValueError("Paper classification head depth is 2 or 4")
        self.convolutions = nn.ModuleList([nn.Conv1d(width if index == 0 else hidden,
                                                    hidden, 3, padding=1) for index in range(depth)])
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, states, mask):
        states = states.float().transpose(1, 2)
        weights = mask.unsqueeze(1).to(states.dtype)
        for convolution in self.convolutions:
            states = F.gelu(convolution(states * weights)) * weights
        pooled = states.sum(-1) / weights.sum(-1).clamp_min(1)
        return self.classifier(pooled)


class EwenModel(nn.Module):
    def __init__(self, tokenizer: nn.Module, backbone: Optional[nn.Module] = None,
                 backbone_name: Optional[str] = None, num_classes: Optional[int] = None,
                 classifier_layers: int = 2, mask_ratio: float = .2,
                 use_covariates: bool = True, use_geometry: bool = True,
                 adapter_mode: str = "structured", residual_rank: int = 16,
                 covariate_loss_weight: float = .1, alignment_width: int = 128,
                 attention_heads: Optional[int] = None, max_time_patches: int = 64,
                 max_channels: int = 256, backbone_kwargs: Optional[dict] = None):
        super().__init__()
        if backbone is None:
            if not backbone_name:
                raise ValueError("Supply a real pretrained backbone or an explicit TinyBackbone")
            from transformers import AutoModelForCausalLM

            backbone = AutoModelForCausalLM.from_pretrained(backbone_name, **(backbone_kwargs or {}))
        self.backbone = backbone
        self.vq = tokenizer
        if not hasattr(tokenizer, "n_embed") or not hasattr(tokenizer, "get_codebook_indices"):
            raise TypeError("EEG tokenizer must expose n_embed and get_codebook_indices")
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        self.vq.eval()
        self.width = int(text_config(backbone).hidden_size)
        self.adapter_mode = adapter_mode
        self.adapter_report = attach_structured_adapters(backbone, residual_rank, adapter_mode)


        codebook = self._tokenizer_codebook()
        code_width = codebook.shape[1] if codebook is not None else min(128, self.width)
        self.code_embedding = (nn.Embedding(int(tokenizer.n_embed), code_width)
                               if codebook is None else None)
        self.eeg_embedding = nn.Linear(code_width, self.width)
        nn.init.normal_(self.eeg_embedding.weight, std=.02)
        nn.init.zeros_(self.eeg_embedding.bias)
        self.covariate = CovariateEncoder(self.width)
        bottleneck = min(128, self.width)
        self.prefix_projection = nn.Sequential(nn.Linear(self.width, bottleneck), nn.GELU(),
                                               nn.Linear(bottleneck, self.width))
        heads = attention_heads or next(value for value in (8, 4, 2, 1) if self.width % value == 0)
        self.geometry = TokenGeometry(self.width, heads, max_time_patches, max_channels)
        self.eeg_attention = EEGAttention(self.width, heads)
        self.descriptor_alignment = nn.Linear(self.width, alignment_width)
        self.neural_alignment = nn.Linear(self.width, alignment_width)
        self.classification_head = (TemporalClassifier(self.width, num_classes, classifier_layers)
                                    if num_classes is not None else None)
        self.mask_ratio = float(mask_ratio)
        if not 0 <= self.mask_ratio < 1:
            raise ValueError("EEG input mask ratio must lie in [0,1)")
        self.use_covariates, self.use_geometry = use_covariates, use_geometry
        self.covariate_loss_weight = float(covariate_loss_weight)

    def train(self, mode=True):
        super().train(mode)
        self.vq.eval()
        return self

    def core_parameters(self):
        return core_parameters(self.backbone)

    def residual_parameters(self):
        return residual_parameters(self.backbone)

    def eeg_parameters(self):
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        return [parameter for parameter in self.parameters() if parameter.requires_grad and
                id(parameter) not in backbone_ids]

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def _backbone_dtype(self):
        return self.backbone.get_input_embeddings().weight.dtype

    def _tokenizer_codebook(self):
        if hasattr(self.vq, "quantizer") and hasattr(self.vq.quantizer, "codebook"):
            return self.vq.quantizer.codebook
        if hasattr(self.vq, "quantize") and hasattr(self.vq.quantize, "embedding"):
            return self.vq.quantize.embedding.weight
        return None

    def parameter_counts(self):

        total = sum(parameter.numel() for parameter in self.parameters())
        tokenizer = sum(parameter.numel() for parameter in self.vq.parameters())
        tokenizer_buffers = sum(buffer.numel() for buffer in self.vq.buffers())
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        interface = sum(parameter.numel() for parameter in self.parameters()
                        if id(parameter) not in backbone_ids and parameter.requires_grad)
        core_count = sum(parameter.numel() for parameter in self.core_parameters())
        residual_count = sum(parameter.numel() for parameter in self.residual_parameters())
        base_count = sum(parameter.numel() for parameter in self.backbone.parameters()) - core_count - residual_count
        return {"total_parameters": total, "frozen_tokenizer_parameters": tokenizer,
                "frozen_tokenizer_buffers": tokenizer_buffers,
                "backbone_base_parameters": base_count,
                "eeg_interface_trainable_parameters": interface,
                "eeg_interface_percent_of_backbone": 100. * interface / max(base_count, 1),
                "all_eeg_tensors_percent_of_backbone": 100. * (interface + tokenizer + tokenizer_buffers) / max(base_count, 1),
                "core_parameters": core_count,
                "residual_parameters": residual_count,
                "total_trainable_parameters": sum(parameter.numel() for parameter in self.trainable_parameters())}

    def build_eeg_embeddings(self, eeg, input_chans, input_times, input_mask,
                             descriptors, modality_mask, descriptor_mask=None,
                             information_condition="full"):
        if information_condition not in {"full", "waveform_only", "descriptor_only", "prompt_only"}:
            raise ValueError(f"Unknown inference information condition: {information_condition}")
        if information_condition != "full" and torch.is_grad_enabled():
            raise ValueError("Information conditions are inference-only; training requires full inputs")
        batch = eeg.shape[0]
        if information_condition == "prompt_only":
            empty = self.eeg_embedding.weight.new_empty(batch, 0, self.width).to(self._backbone_dtype())
            mask = torch.zeros(batch, 0, dtype=torch.bool, device=empty.device)
            return empty, mask, self.eeg_embedding.weight.sum() * 0, 0, 0
        if information_condition == "descriptor_only":
            if not self.use_covariates:
                raise ValueError("Descriptor-only inference requires an enabled covariate pathway")
            covariate, _ = self.covariate(descriptors, modality_mask, descriptor_mask)
            prefix = self.prefix_projection(covariate).unsqueeze(1).to(self._backbone_dtype())
            return prefix, torch.ones(batch, 1, dtype=torch.bool, device=prefix.device), covariate.sum() * 0, 0, 1
        valid = input_mask.bool()
        if not valid.any(1).all():
            raise ValueError("Every EEG window must contain at least one valid token")

        with torch.no_grad():
            self.vq.eval()
            codes = self.vq.get_codebook_indices(eeg, input_chans, input_times, valid)
        if isinstance(codes, (tuple, list)):
            codes = codes[0]
        if codes.shape != valid.shape:
            raise ValueError("Tokenizer indices must preserve channel/time token grid")
        codebook = self._tokenizer_codebook()
        vectors = (self.code_embedding(codes.long()) if codebook is None else
                   F.embedding(codes.long(), codebook).float())
        states = self.eeg_embedding(vectors)
        neural_pooled = masked_pool(states, valid)
        covariates_enabled = self.use_covariates and information_condition != "waveform_only"
        if covariates_enabled:
            covariate, descriptor_valid = self.covariate(descriptors, modality_mask, descriptor_mask)
            cov_windows = descriptor_valid[:, :7].any(1) & valid.any(1)
            differences = self.descriptor_alignment(covariate) - self.neural_alignment(neural_pooled)
            cov_loss = differences.square().sum(-1)[cov_windows].mean() if cov_windows.any() else states.sum() * 0
            conditioned = self.covariate.apply_modulation(states, covariate)
        else:
            covariate = states.new_zeros(batch, self.width)
            cov_loss = states.sum() * 0
            conditioned = states
            cov_windows = torch.zeros(batch, dtype=torch.bool, device=states.device)
        position, bias = self.geometry(input_times, input_chans, covariate, valid, self.use_geometry)
        conditioned = conditioned + position
        if self.training and self.mask_ratio:
            hidden_tokens = (torch.rand(valid.shape, device=valid.device) < self.mask_ratio) & valid
            conditioned = conditioned.masked_fill(hidden_tokens.unsqueeze(-1), 0)
        eeg_states = self.eeg_attention(conditioned, valid, bias)
        if covariates_enabled:
            prefix = self.prefix_projection(covariate).unsqueeze(1)
            embeddings = torch.cat((prefix, eeg_states), dim=1)
            attention_mask = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)
            prefix_count = 1
        else:
            embeddings, attention_mask, prefix_count = eeg_states, valid, 0
        return embeddings.to(self._backbone_dtype()), attention_mask, cov_loss, int(cov_windows.sum()), prefix_count

    def _decoder(self, embeddings, attention_mask, hidden=False):

        positions = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        return self.backbone(inputs_embeds=embeddings, attention_mask=attention_mask.long(),
                             position_ids=positions, use_cache=False, output_hidden_states=hidden,
                             return_dict=True)

    def forward_classification(self, eeg, input_chans, input_times, input_mask,
                               descriptors, modality_mask, labels=None, descriptor_mask=None,
                               label=None, **unused):
        if self.classification_head is None:
            raise ValueError("Classification head was not configured")
        if labels is None:
            labels = label
        embeddings, mask, cov_loss, cov_count, prefix = self.build_eeg_embeddings(
            eeg, input_chans, input_times, input_mask, descriptors, modality_mask, descriptor_mask)
        output = self._decoder(embeddings, mask, hidden=True)
        hidden = output.hidden_states[-1][:, prefix:]


        maximum = int(input_times[input_mask.bool()].max()) + 1
        pooled_time = hidden.new_zeros(hidden.shape[0], maximum, hidden.shape[-1])
        counts = hidden.new_zeros(hidden.shape[0], maximum, 1)
        safe_times = input_times.masked_fill(~input_mask.bool(), 0)
        scatter_index = safe_times.unsqueeze(-1).expand_as(hidden)
        pooled_time.scatter_add_(1, scatter_index, hidden * input_mask.unsqueeze(-1))
        counts.scatter_add_(1, safe_times.unsqueeze(-1), input_mask.unsqueeze(-1).to(hidden.dtype))
        pooled_time = pooled_time / counts.clamp_min(1)
        logits = self.classification_head(pooled_time, counts.squeeze(-1) > 0)
        loss = (F.cross_entropy(logits, labels.long()) + self.covariate_loss_weight * cov_loss
                if labels is not None else None)
        return EwenOutput(loss=loss, logits=logits, cov_loss=cov_loss,
                          valid_covariate_windows=cov_count, hidden_states=hidden)

    @staticmethod
    def _token_loss(logits, labels):
        shifted_labels = labels[:, 1:]
        count = int((shifted_labels != -100).sum())
        if count == 0:
            raise ValueError("No valid next-token targets in the supplied batch")
        total = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                shifted_labels.reshape(-1), ignore_index=-100, reduction="sum")
        return total / count, total, count

    def forward_generation(self, eeg, input_chans, input_times, input_mask,
                           descriptors, modality_mask, prompt_ids, target_ids,
                           prompt_attention_mask=None, target_attention_mask=None,
                           descriptor_mask=None, **unused):
        eeg_embeddings, eeg_mask, cov_loss, cov_count, _ = self.build_eeg_embeddings(
            eeg, input_chans, input_times, input_mask, descriptors, modality_mask, descriptor_mask)
        prompt_mask = (torch.ones_like(prompt_ids, dtype=torch.bool) if prompt_attention_mask is None
                       else prompt_attention_mask.bool())
        target_mask = (torch.ones_like(target_ids, dtype=torch.bool) if target_attention_mask is None
                       else target_attention_mask.bool())

        text_embedding = self.backbone.get_input_embeddings()
        sequences, masks, labels_list = [], [], []
        for row in range(eeg.shape[0]):
            eeg_valid = eeg_embeddings[row, eeg_mask[row]]
            prompt = prompt_ids[row, prompt_mask[row]]
            target = target_ids[row, target_mask[row]]
            sequence = torch.cat((eeg_valid, text_embedding(prompt), text_embedding(target)), dim=0)
            labels = torch.full((sequence.shape[0],), -100, device=sequence.device, dtype=torch.long)
            labels[eeg_valid.shape[0] + prompt.shape[0]:] = target
            sequences.append(sequence)
            masks.append(torch.ones(sequence.shape[0], device=sequence.device, dtype=torch.bool))
            labels_list.append(labels)
        embeddings = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        attention_mask = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=False)
        labels = nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        output = self._decoder(embeddings, attention_mask)
        generation_loss, loss_sum, count = self._token_loss(output.logits, labels)
        loss = generation_loss + self.covariate_loss_weight * cov_loss
        return EwenOutput(loss=loss, logits=output.logits, cov_loss=cov_loss,
                          generation_loss=generation_loss, generation_loss_sum=loss_sum,
                          generation_token_count=count, valid_target_tokens=count,
                          valid_covariate_windows=cov_count)

    def forward_text(self, text_ids=None, text_attention_mask=None, input_ids=None,
                     attention_mask=None, **unused):
        ids = text_ids if text_ids is not None else input_ids
        valid = text_attention_mask if text_attention_mask is not None else attention_mask
        if ids is None:
            raise ValueError("Text-only inputs require token IDs")
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid.bool()
        if ((valid[:, 1:]) & (~valid[:, :-1])).any():
            raise ValueError("Text-only scoring requires contiguous right-padded token sequences")
        labels = ids.masked_fill(~valid, -100)
        output = self._decoder(self.backbone.get_input_embeddings()(ids), valid)
        mean, total, count = self._token_loss(output.logits, labels)
        return EwenOutput(loss=mean, logits=output.logits, generation_loss=mean,
                          generation_loss_sum=total, generation_token_count=count,
                          valid_target_tokens=count)

    def forward(self, mode="generation", **kwargs):
        routes = {"classification": self.forward_classification,
                  "generation": self.forward_generation, "text": self.forward_text}
        if mode not in routes:
            raise ValueError(f"Unknown forward mode: {mode}")
        return routes[mode](**kwargs)

    @torch.no_grad()
    def generate(self, eeg, input_chans, input_times, input_mask, descriptors, modality_mask,
                 prompt_ids, prompt_attention_mask=None, descriptor_mask=None,
                 max_new_tokens=128, eos_token_id=None, temperature=0.0, top_p=1.0,
                 generator=None, information_condition="full", **unused):

        if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid autoregressive decoding configuration")
        original_training = self.training
        self.eval()
        try:
            embeddings, eeg_mask, _, _, _ = self.build_eeg_embeddings(
                eeg, input_chans, input_times, input_mask, descriptors, modality_mask, descriptor_mask,
                information_condition=information_condition)
            prompt_mask = (torch.ones_like(prompt_ids, dtype=torch.bool) if prompt_attention_mask is None
                           else prompt_attention_mask.bool())
            eos = eos_token_id if eos_token_id is not None else getattr(text_config(self.backbone), "eos_token_id", None)
            eos_set = set(eos) if isinstance(eos, (tuple, list)) else ({eos} if eos is not None else set())
            outputs = []
            embed = self.backbone.get_input_embeddings()
            for row in range(embeddings.shape[0]):
                sequence = torch.cat((embeddings[row, eeg_mask[row]], embed(prompt_ids[row, prompt_mask[row]])), dim=0)
                if sequence.shape[0] == 0:
                    raise ValueError("Autoregressive inference requires a nonempty prompt or conditioning prefix")
                tokens = []
                for _ in range(max_new_tokens):
                    output = self._decoder(sequence.unsqueeze(0), torch.ones(1, sequence.shape[0],
                                                                            dtype=torch.bool, device=sequence.device))
                    logits = output.logits[0, -1].float()
                    if temperature == 0:
                        token = logits.argmax()
                    else:
                        probabilities = (logits / temperature).softmax(-1)
                        if top_p < 1:
                            sorted_probabilities, indices = probabilities.sort(descending=True)
                            discard = sorted_probabilities.cumsum(-1) - sorted_probabilities >= top_p
                            sorted_probabilities = sorted_probabilities.masked_fill(discard, 0)
                            sorted_probabilities /= sorted_probabilities.sum()
                            token = indices[torch.multinomial(sorted_probabilities, 1, generator=generator)].squeeze(0)
                        else:
                            token = torch.multinomial(probabilities, 1, generator=generator).squeeze(0)
                    tokens.append(token)
                    if int(token) in eos_set:
                        break
                    sequence = torch.cat((sequence, embed(token.reshape(1))), dim=0)
                outputs.append(torch.stack(tokens))
            return outputs
        finally:
            self.train(original_training)


class _TinyAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [nn.Linear(width, width) for _ in range(4)]

    def forward(self, states, mask):
        batch, length, width = states.shape
        shape = (batch, length, self.heads, width // self.heads)
        query, key, value = [projection(states).reshape(shape).transpose(1, 2)
                             for projection in (self.q_proj, self.k_proj, self.v_proj)]
        causal = torch.ones(length, length, device=states.device, dtype=torch.bool).tril()
        permitted = causal[None, None] & mask[:, None, None, :].bool()
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=permitted)
        return self.o_proj(result.transpose(1, 2).reshape(batch, length, width))


class _TinyMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(width, 2 * width), nn.Linear(width, 2 * width)
        self.down_proj = nn.Linear(2 * width, width)

    def forward(self, states):
        return self.down_proj(F.silu(self.gate_proj(states)) * self.up_proj(states))


class _TinyLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attn, self.mlp = _TinyAttention(width, heads), _TinyMLP(width)
        self.attention_norm, self.mlp_norm = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, states, mask):
        states = states + self.self_attn(self.attention_norm(states), mask)
        return states + self.mlp(self.mlp_norm(states))


class TinyBackbone(nn.Module):


    def __init__(self, vocab_size=128, hidden_size=32, layers=2, heads=4, max_positions=4096):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, layer_types=["full_attention"] * layers,
                                      eos_token_id=2, vocab_size=vocab_size)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.positions = nn.Embedding(max_positions, hidden_size)
        self.model.layers = nn.ModuleList([_TinyLayer(hidden_size, heads) for _ in range(layers)])
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, output_hidden_states=False, **unused):
        embeddings = inputs_embeds if inputs_embeds is not None else self.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.bool)
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        states = embeddings + self.model.positions(position_ids)
        hidden_states = [states]
        for layer in self.model.layers:
            states = layer(states, attention_mask)
            hidden_states.append(states)
        states = self.model.norm(states)
        hidden_states[-1] = states
        return SimpleNamespace(logits=self.lm_head(states),
                               hidden_states=tuple(hidden_states) if output_hidden_states else None)
