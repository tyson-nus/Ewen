from __future__ import annotations
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict
import math
import random
import numpy as np
from pathlib import Path
import torch
from .adapters import core_parameters, parameter_groups
from .projection import project_core_gradients

def autocast_for(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for (key, value) in batch.items()}

def eeg_arguments(batch):
    names = ('eeg', 'input_chans', 'input_times', 'input_mask', 'descriptors', 'descriptor_mask', 'modality_mask')
    return {name: batch[name] for name in names if name in batch}

def encode_answers(tokenizer, prompts, targets, device, max_prompt_tokens=128, max_target_tokens=768):
    if any((not isinstance(target, str) or not target.strip() for target in targets)):
        raise ValueError('Generation target is empty')
    prompt = tokenizer(prompts, padding=True, truncation=False, return_tensors='pt', add_special_tokens=True)
    target = tokenizer(targets, padding=True, truncation=False, return_tensors='pt', add_special_tokens=False)
    eos = getattr(tokenizer, 'eos_token_id', None)
    if eos is None:
        raise ValueError('Autoregressive training requires an explicit EOS token')
    complete = []
    for (row, mask) in zip(target['input_ids'], target['attention_mask']):
        ids = row[mask.bool()]
        if not len(ids) or int(ids[-1]) != int(eos):
            ids = torch.cat((ids, ids.new_tensor([eos])))
        complete.append(ids)
    ids = torch.nn.utils.rnn.pad_sequence(complete, batch_first=True, padding_value=tokenizer.pad_token_id)
    masks = torch.nn.utils.rnn.pad_sequence([torch.ones_like(row) for row in complete], batch_first=True, padding_value=0)
    target = {'input_ids': ids, 'attention_mask': masks}
    if int(prompt['attention_mask'].sum(1).max()) > max_prompt_tokens:
        raise ValueError('Prompt exceeds configured token budget')
    if int(target['attention_mask'].sum(1).max()) > max_target_tokens:
        raise ValueError('Target exceeds configured token budget; complete descriptions are required')
    if torch.any(target['attention_mask'].sum(1) == 0):
        raise ValueError('Generation target is empty')
    return {'prompt_ids': prompt['input_ids'].to(device), 'prompt_attention_mask': prompt['attention_mask'].to(device), 'target_ids': target['input_ids'].to(device), 'target_attention_mask': target['attention_mask'].to(device)}

def make_optimizer(model, lr, weight_decay=0.1):
    groups = parameter_groups(model, lr, weight_decay)
    groups = [group for group in groups if group['params']]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))

def make_scheduler(optimizer, total_steps, warmup_fraction=0.03, minimum_ratio=0.1):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, rate)

def globally_weighted(mean_loss, local_count):
    count = torch.as_tensor(local_count, dtype=torch.float64, device=mean_loss.device)
    total = global_sum(count)
    if float(total) <= 0:
        raise ValueError('No valid supervised observations in distributed step')
    return (mean_loss * (count / total).to(mean_loss.dtype), float(total))

def online_step(model, optimizer, batch, text_tokenizer, owt_stream, *, task='classification', lambda_cov=0.1, reference_k=4, reference_batch=8, reference_length=512, rho=0.5, apply_projection=True, clip_norm=1.0, max_target_tokens=768):
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    model.train()
    args = eeg_arguments(batch)
    with autocast_for(device):
        if task == 'classification':
            output = model.forward_classification(**args, labels=batch['labels'])
            primary = output.loss - model.covariate_loss_weight * output.cov_loss
            local_count = len(batch['labels'])
        else:
            answers = encode_answers(text_tokenizer, batch['prompts'], batch['targets'], device, max_target_tokens=max_target_tokens)
            output = model.forward_generation(**args, **answers)
            primary = output.generation_loss
            local_count = output.generation_token_count
        (task_mean, global_count) = globally_weighted(primary, local_count)
        cov_count = torch.tensor(float(output.valid_covariate_windows), device=device)
        global_cov_count = global_sum(cov_count)
        cov_mean = output.cov_loss * cov_count / global_cov_count.clamp_min(1)
        loss = task_mean + lambda_cov * cov_mean
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite task objective; optimizer step aborted')
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    sum_gradients(parameters)
    cores = core_parameters(model)
    diagnostics = None
    if apply_projection and cores:
        if owt_stream is None:
            raise ValueError('Projected adaptation requires actual OpenWebText token stream')
        references = []
        for _ in range(reference_k):
            (ids, mask) = owt_stream.batch(reference_batch, reference_length, device)
            with autocast_for(device):
                text_output = model.forward_text(ids, mask)
                (language_mean, _) = globally_weighted(text_output.loss, int(mask[:, 1:].sum()))
            gradients = torch.autograd.grad(language_mean, cores, allow_unused=True)
            reduced = []
            for (parameter, grad) in zip(cores, gradients):
                grad = torch.zeros_like(parameter) if grad is None else grad.detach().float()
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(grad)
                reduced.append(grad)
            references.append(reduced)
        diagnostics = project_core_gradients(cores, references, rho=rho, expected_k=reference_k)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    logged_primary = float(global_sum(task_mean.detach()).float())
    logged_cov = float(global_sum(cov_mean.detach()).float())
    return {'loss': logged_primary + lambda_cov * logged_cov, 'primary_loss': logged_primary, 'cov_loss': logged_cov, 'global_supervision_count': global_count, 'supplied_gradient_norm': float(grad_norm), 'projection': asdict(diagnostics) if diagnostics else None}

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, manifest, runtime_state=None):
    names = {name for (name, parameter) in model.named_parameters() if parameter.requires_grad}
    state = {name: tensor.detach().cpu() for (name, tensor) in model.state_dict().items() if name in names}
    payload = {'format': 'ewen-adaptation-v1', 'trainable_state': state, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler is not None else None, 'epoch': epoch, 'step': step, 'manifest': manifest, 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'runtime_state': runtime_state or {}}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(destination)

def load_adaptation(path, model, optimizer=None, scheduler=None, expected_identity=None, restore_rng=False):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'ewen-adaptation-v1':
        raise ValueError('Unknown adaptation checkpoint format')
    if expected_identity is not None and payload['manifest'].get('resource_identity') != expected_identity:
        raise ValueError('Checkpoint base/data/tokenizer/statistics identity differs from current experiment')
    expected = {name for (name, parameter) in model.named_parameters() if parameter.requires_grad}
    if set(payload['trainable_state']) != expected:
        raise ValueError('Checkpoint mutable tensor set differs from configured experiment')
    (missing, unexpected) = model.load_state_dict(payload['trainable_state'], strict=False)
    if unexpected or expected.intersection(missing):
        raise ValueError('Adaptation checkpoint fails exact mutable tensor coverage')
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if scheduler is not None and payload['scheduler'] is not None:
        scheduler.load_state_dict(payload['scheduler'])
    if restore_rng:
        torch.set_rng_state(payload['rng'])
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        if torch.cuda.is_available() and payload['cuda_rng']:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload

def load_stage_initialization(path, model, expected_identity):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'ewen-adaptation-v1' or payload.get('manifest', {}).get('task') != 'pretraining':
        raise ValueError('Stage initialization requires an Ewen next-token pretraining checkpoint')
    source = payload['manifest'].get('resource_identity', {})
    for key in ('base', 'vq_sha256'):
        if source.get(key) != expected_identity.get(key):
            raise ValueError('Pretraining checkpoint immutable resource differs: ' + key)
    for key in ('adapter_mode', 'residual_rank'):
        if source.get('model_semantics', {}).get(key) != expected_identity.get('model_semantics', {}).get(key):
            raise ValueError('Pretraining checkpoint adapter schema differs: ' + key)
    expected = {name for (name, parameter) in model.named_parameters() if parameter.requires_grad}
    new_head = {name for name in expected if name.startswith('classification_head.')}
    if set(payload['trainable_state']) != expected - new_head:
        raise ValueError('Pretraining checkpoint lacks exact shared mutable tensor coverage')
    current = model.state_dict()
    for (name, tensor) in payload['trainable_state'].items():
        if tensor.shape != current[name].shape:
            raise ValueError('Pretraining shared tensor shape differs: ' + name)
    (missing, unexpected) = model.load_state_dict(payload['trainable_state'], strict=False)
    if unexpected or (expected - new_head).intersection(missing):
        raise ValueError('Pretraining shared tensor loading failed')
    return payload

def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ByteTokenizer:
    (pad_token_id, bos_token_id, eos_token_id) = (0, 1, 2)

    def __call__(self, texts, padding=True, truncation=False, max_length=None, return_tensors='pt', **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [[self.bos_token_id] + [b + 3 for b in text.encode()] + [self.eos_token_id] for text in texts]
        if truncation and max_length:
            rows = [row[:max_length] for row in rows]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for (i, row) in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return {'input_ids': ids, 'attention_mask': mask}

    def decode(self, ids, **kwargs):
        return bytes((int(i) - 3 for i in ids if 3 <= int(i) <= 258)).decode(errors='replace')

class OWTStream:

    def __init__(self, path, vocab_size, seed=13, train_fraction=0.9, tokenizer_identity=None):
        path = require_path(path, 'owt_tokens')
        if tokenizer_identity:
            provenance = json.loads(Path(path).with_suffix('.provenance.json').read_text())
            if provenance.get('source_split') != 'train':
                raise ValueError('Language projection must use the declared training split')
            recorded = provenance.get('tokenizer_sha256', provenance.get('tokenizer_files_sha256', {}).get('tokenizer.json'))
            if recorded != tokenizer_identity:
                raise ValueError('OWT token stream was encoded with a different tokenizer')
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                while (block := stream.read(1 << 20)):
                    digest.update(block)
            if digest.hexdigest() != provenance.get('target_sha256'):
                raise ValueError('OWT token resource differs from its conversion provenance')
        self.tokens = np.memmap(path, mode='r', dtype=np.uint32)
        if len(self.tokens) < 4096:
            raise ValueError('OWT stream is too short')
        self.train_end = int(len(self.tokens) * train_fraction)
        self.vocab_size = vocab_size
        self.generator = np.random.default_rng(seed)
        for start in range(0, len(self.tokens), 1 << 24):
            if self.tokens[start:start + (1 << 24)].max() >= vocab_size:
                raise ValueError('OWT token ids are incompatible with backbone vocabulary')
        self.identity = {'token_count': len(self.tokens), 'train_end': self.train_end, 'dtype': 'uint32', 'size_bytes': Path(path).stat().st_size}

    def batch(self, count, length, device, evaluation=False, offset=None):
        if count < 1 or length < 2:
            raise ValueError('OWT batches require positive count and at least two tokens')
        if offset is not None and (not isinstance(offset, int) or offset < 0):
            raise ValueError('OWT fixed offsets must be nonnegative integers within their partition')
        (lo, hi) = (self.train_end, len(self.tokens)) if evaluation else (0, self.train_end)
        if hi - lo <= length:
            raise ValueError('OWT partition is shorter than configured context')
        if offset is None:
            starts = self.generator.integers(lo, hi - length, count)
        else:
            starts = lo + np.arange(count) * length + offset
            if np.max(starts) + length > hi:
                raise ValueError('Evaluation exceeds held-out OWT range')
        ids = torch.from_numpy(np.stack([np.asarray(self.tokens[s:s + length], dtype=np.int64) for s in starts])).to(device)
        return (ids, torch.ones_like(ids))

def distributed_context():
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if world > 1 and (not torch.distributed.is_initialized()):
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        torch.distributed.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    return (rank, world, device)

def global_sum(value):
    value = value.clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return value

def sum_gradients(parameters):
    if not torch.distributed.is_initialized():
        return
    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters:
        return
    present = torch.tensor([parameter.grad is not None for parameter in parameters], dtype=torch.int32, device=parameters[0].device)
    torch.distributed.all_reduce(present, op=torch.distributed.ReduceOp.MAX)
    for (parameter, globally_present) in zip(parameters, present.tolist()):
        if not globally_present:
            parameter.grad = None
            continue
        grad = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
        parameter.grad = grad

def require_path(value, field):
    if value is None or (isinstance(value, str) and (not value.strip())):
        raise ValueError(f"Set '{field}' before running this command")
    return value
