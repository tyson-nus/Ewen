from __future__ import annotations
DEFAULT_CONFIG = {'variant': 'S', 'adapter_mode': 'structured', 'reference_k': 4, 'rho': 0.5, 'lambda_cov': 0.1, 'residual_rank': 16, 'mask_ratio': 0.2, 'seed': 13, 'epochs_classification': 50, 'epochs_pretraining': 10, 'epochs_generation': 10, 'reference_batch': 8, 'reference_length': 512, 'learning_rate': 2e-05, 'weight_decay': 0.1, 'warmup_fraction': 0.03, 'minimum_lr_ratio': 0.1, 'clip_norm': 1.0, 'max_prompt_tokens': 128, 'max_target_tokens': 768, 'max_new_tokens': 768, 'field_prompts': ['label', 'relation', 'band', 'summary'], 'target_policy': 'independent_waveform_or_annotation', 'use_covariates': True, 'use_geometry': True, 'classification_head_layers': {'default': 2, 'SHU-MI': 4, 'mumtaz': 4}, 'classification_learning_rates': {'default': 2e-05, 'SHU-MI': 4e-05}, 'implementation_choices': {'seeds': 'Explicit reconstruction seeds; the manuscript does not specify identities.', 'reference_length': '512-token context; exact OWT context is absent from the manuscript.', 'token_limits': 'Complete targets are required; over-budget records abort instead of truncation.'}, 'data_root': '', 'manifest_path': '', 'vq_checkpoint': '', 'backbone_path': '', 'owt_tokens': '', 'text_model': '', 'targets_path': '', 'variant_backbone_paths': {'S': '', 'B': '', 'L': ''}, 'variant_owt_tokens': {'S': '', 'B': '', 'L': ''}, 'pretraining_checkpoints': {}, 'method_pretraining_checkpoints': {}}
import copy
import hashlib
import os
import sys
from .data import require_paths
from .training import seed_everything, ByteTokenizer, OWTStream, distributed_context
import argparse
from dataclasses import asdict
import gc
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .data import H5EEGDataset, PAPER, DATASET_NAMES, collate, audit_dataset, build_paper_index
from .descriptors import DescriptorStandardizer
from .model import EwenModel, TinyBackbone
from .tokenizer import load_tokenizer, sha256_file, PaperVQTokenizer, EncoderConfig
from .training import online_step, make_optimizer, make_scheduler, move_batch, eeg_arguments, autocast_for, save_checkpoint, load_adaptation, load_stage_initialization
ROOT = Path(__file__).resolve().parents[1]
FIELD_PROMPTS = {'label': 'Name the dataset task class supported by this EEG window.', 'band': 'Which EEG frequency band has the greatest relative power in this window? Answer with one band name.', 'relation': 'Describe the relationship between the available auxiliary physiology and EEG. If auxiliary measurements are absent, state that the relationship is unavailable.', 'summary': "Describe this EEG window's spectral characteristics, available physiological relationships, and temporal and spatial variation supported by the waveform."}
FIELD_PARAPHRASES = {'label': [FIELD_PROMPTS['label'], 'Identify the task category evidenced by this EEG window.', "State the EEG window's class name."], 'band': [FIELD_PROMPTS['band'], 'Name the frequency band with the largest relative EEG power.', 'Identify the dominant spectral band of this EEG segment.'], 'relation': [FIELD_PROMPTS['relation'], 'Explain the measured EEG and auxiliary physiological relationship, indicating when evidence is unavailable.', 'Report any supported relationship with EOG, ECG or EMG; state unavailability when those measurements are absent.'], 'summary': [FIELD_PROMPTS['summary'], "Summarize the spectral and physiological evidence and the waveform's spatial and temporal variation.", 'Write an evidence-based description of this EEG segment, including available auxiliary physiology and changes across time and channels.']}
FIELD_PREFIXES = {field: ['Answer: ', 'Response: ', 'Report: ', 'Interpretation: ', 'Assessment: ', 'Finding: '] for field in FIELD_PROMPTS}

class GenerationFields(torch.utils.data.Dataset):

    def __init__(self, base, fields, seed=13):
        if not fields or len(set(fields)) != len(fields) or any((field not in FIELD_PROMPTS for field in fields)):
            raise ValueError('Generation fields must be distinct declared task names')
        (self.base, self.fields) = (base, tuple(fields))
        (self.seed, self.epoch) = (seed, 0)

    def __len__(self):
        return len(self.base) * len(self.fields)

    def __getitem__(self, index):
        row = dict(self.base[index // len(self.fields)])
        field = self.fields[index % len(self.fields)]
        target = row['targets'].get(field)
        if not target:
            raise ValueError(f'Independent paired target is missing for field {field}')
        choice = int(digest_json([self.seed, self.epoch, index, field])[:16], 16)
        row['prompt'] = FIELD_PARAPHRASES[field][choice % len(FIELD_PARAPHRASES[field])] + ' Dataset: ' + row['dataset_name'] + '.'
        row['target'] = FIELD_PREFIXES[field][choice // len(FIELD_PARAPHRASES[field]) % len(FIELD_PREFIXES[field])] + target
        return row

def dataset(config, names, split, *, limit=None, scaler=None, diagnostic=False, templates=False):
    return H5EEGDataset(config['data_root'], names, split, max_samples=limit, sfreq_overrides=config.get('sfreq_overrides'), manifest_path=config.get('manifest_path'), standardizer=scaler, allow_audited_mismatch=diagnostic, channel_identity=config.get('channel_identity', 'stored'), notch_hz=config.get('notch_hz'), targets_path=config.get('targets_path'), smoke_template_targets=templates)

def model_and_text(config, device, classes=None, layers=2, tiny=False):
    require_paths(config, 'vq_checkpoint')
    if not tiny:
        require_paths(config, 'backbone_path')
    vq = load_tokenizer(config['vq_checkpoint'])
    if tiny:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(13)
            backbone = TinyBackbone(vocab_size=259, hidden_size=32, layers=2)
        text = ByteTokenizer()
        model = EwenModel(vq, backbone=backbone, num_classes=classes, classifier_layers=layers, adapter_mode=config['adapter_mode'], mask_ratio=config['mask_ratio'], use_covariates=config['use_covariates'], use_geometry=config['use_geometry'])
    else:
        from transformers import AutoTokenizer
        text = AutoTokenizer.from_pretrained(config['backbone_path'], local_files_only=True)
        if text.pad_token_id is None:
            text.pad_token = text.eos_token
        model = EwenModel(vq, backbone_name=config['backbone_path'], num_classes=classes, classifier_layers=layers, adapter_mode=config['adapter_mode'], mask_ratio=config['mask_ratio'], use_covariates=config['use_covariates'], use_geometry=config['use_geometry'], residual_rank=config['residual_rank'], backbone_kwargs={'local_files_only': True, 'dtype': torch.bfloat16 if device.type == 'cuda' and config['adapter_mode'] != 'full' else torch.float32})
    return (model.to(device), text)

def protocol_manifest(config, data, kind, diagnostic=False):
    return {'schema': 'ewen-experiment-v1', 'artifact_kind': kind, 'diagnostic': diagnostic, 'paper_equivalence': False if diagnostic else 'requires_protocol_review', 'config': public_config(config), 'environment': environment(), 'data_audits': data.audits, 'sample_count': len(data), 'sample_ids_hash': digest_json([row['sample_key'] for row in data.records]), 'implementation_sha256': {path.name: sha256_file(path) for path in sorted((ROOT / 'ewen_repro').glob('*.py'))}}

def resource_identity(config, model, data, statistics_path, tiny=False):
    language_hash = None
    if model.core_parameters():
        require_paths(config, 'owt_tokens')
        language_hash = sha256_file(config['owt_tokens'])
    base = {'type': 'tiny-diagnostic', 'seed': 13, 'config': vars(model.backbone.config)} if tiny else {'config_sha256': sha256_file(Path(config['backbone_path']) / 'config.json'), 'weight_sha256': {path.name: sha256_file(path) for path in sorted(Path(config['backbone_path']).glob('*.safetensors'))}, 'tokenizer_sha256': {path.name: sha256_file(path) for path in sorted(Path(config['backbone_path']).glob('tokenizer*.json'))}}
    semantics = {key: config[key] for key in ('adapter_mode', 'mask_ratio', 'use_covariates', 'use_geometry', 'residual_rank', 'classification_head_layers')}
    semantics['classifier_output_size'] = model.classification_head.classifier.out_features if model.classification_head is not None else None
    return {'base': base, 'model_semantics': semantics, 'vq_sha256': model.vq.provenance['sha256'], 'statistics_sha256': sha256_file(statistics_path), 'dataset_manifest_hashes': {name: report['manifest_sha256'] for (name, report) in data.audits.items()}, 'language_training_stream_sha256': language_hash}

def command_audit(args, config):
    manifests = config.get('manifest_path')

    def manifest_for(name):
        if isinstance(manifests, dict):
            require_paths(config, 'manifest_path.' + name)
            return manifests[name]
        return str(Path(manifests) / name / 'aligned_samples.jsonl') if manifests else None
    reports = {name: audit_dataset(config['data_root'], name, manifest_path=manifest_for(name), sfreq_overrides=config.get('sfreq_overrides')) for name in args.datasets}
    save_json(args.output, {'artifact_kind': 'data_audit', 'datasets': reports})
    for (name, report) in reports.items():
        print(name, 'PASS' if not report['errors'] else 'PROTOCOL_CONFLICT', '; '.join(report['errors']))

def command_prepare(args, config):
    reports = build_paper_index(config['data_root'], args.output, args.datasets, seed=config['seed'], seedv_subject_sets=config.get('seedv_subject_sets'), sfreq_overrides=config.get('sfreq_overrides'), mumtaz_test_subjects=config.get('mumtaz_test_subjects'))
    save_json(Path(args.output) / 'protocol_audits.json', {'artifact_kind': 'partition_reconstruction', 'datasets': reports})
    print('partition manifests written', len(reports))

def command_statistics(args, config):
    data = dataset(config, args.datasets, 'train', limit=args.limit, diagnostic=args.diagnostic)
    scaler = data.fit_standardizer()
    scaler.save(args.output)
    print('train_only_statistics', scaler.sample_count, 'observed_counts', scaler.count.tolist())
    data.close()

def validation(model, data, device, batch_size):
    model.eval()
    (probabilities, labels) = ([], [])
    with torch.no_grad():
        for raw in DataLoader(data, batch_size=batch_size, collate_fn=collate):
            batch = move_batch(raw, device)
            with autocast_for(device):
                out = model.forward_classification(**eeg_arguments(batch))
            probabilities.append(out.logits.float().softmax(-1).cpu().numpy())
            labels.extend(batch['labels'].cpu().tolist())
    from .metrics import classification_metrics
    return (classification_metrics(labels, np.concatenate(probabilities), model.classification_head.classifier.out_features), np.concatenate(probabilities), labels)

def command_train(args, config):
    (rank, world, device) = distributed_context()
    seed_everything(config['seed'])
    diagnostic = args.diagnostic
    scaler = DescriptorStandardizer.load(args.statistics)
    names = args.datasets
    if args.task == 'classification' and len(names) != 1:
        raise ValueError('Closed-vocabulary paper recipe trains each dataset independently')
    data = dataset(config, names, 'train', limit=args.limit, scaler=scaler, diagnostic=diagnostic, templates=args.task == 'pretraining' or (diagnostic and (args.task != 'generation' or getattr(args, 'smoke_template_targets', False))))
    requested_fields = getattr(args, 'fields', None) or config.get('field_prompts', ['label', 'relation', 'band', 'summary'])
    if args.task == 'generation':
        if not requested_fields or len(set(requested_fields)) != len(requested_fields) or any((field not in FIELD_PROMPTS for field in requested_fields)):
            raise ValueError('Generation fields must be distinct declared task names')
    if args.task == 'generation' and (not diagnostic):
        for i in range(len(data)):
            row = data[i]
            annotation_only = len(requested_fields) == 1 and requested_fields[0] == 'label' and (row['target_provenance'] == 'label_only')
            if any((not isinstance(row['targets'].get(field), str) or not row['targets'][field].strip() for field in requested_fields)) or (not annotation_only and row['target_provenance'] not in {'independent_waveform_measurements', 'independent_annotation'}):
                raise ValueError('Formal generation requires independent waveform-grounded paired descriptions for every training sample')
    classes = PAPER[names[0]]['classes'] if args.task == 'classification' else None
    depth = config['classification_head_layers'].get(names[0], config['classification_head_layers']['default'])
    (model, text) = model_and_text(config, device, classes, depth, args.tiny)
    if args.tiny and (not diagnostic):
        raise ValueError('TinyBackbone only supports explicitly diagnostic execution')
    if args.resume and args.initialize_from:
        raise ValueError('Choose exact resume or pretraining-stage initialization')
    if args.initialize_from:
        load_stage_initialization(args.initialize_from, model, resource_identity(config, model, data, args.statistics, args.tiny))
    if hasattr(model.backbone, 'gradient_checkpointing_enable'):
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    lr = config['classification_learning_rates'].get(names[0], config['learning_rate']) if args.task == 'classification' else config['learning_rate']
    optimizer = make_optimizer(model, lr, config['weight_decay'])
    training_data = GenerationFields(data, requested_fields, config['seed']) if args.task == 'generation' else data
    sampler = DistributedSampler(training_data, num_replicas=world, rank=rank, shuffle=True, seed=config['seed'], drop_last=True) if world > 1 else None
    order_generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(training_data, batch_size=args.batch_size, collate_fn=collate, sampler=sampler, shuffle=sampler is None, num_workers=args.workers, generator=order_generator)
    if not len(loader):
        raise ValueError('Training split has no batches under the configured sampler')
    epoch_key = {'classification': 'epochs_classification', 'pretraining': 'epochs_pretraining', 'generation': 'epochs_generation'}[args.task]
    epochs = args.epochs or config[epoch_key]
    total = min(epochs * len(loader), args.max_steps) if args.max_steps else epochs * len(loader)
    scheduler = make_scheduler(optimizer, total, config['warmup_fraction'], config['minimum_lr_ratio'])
    owt = None
    if model.core_parameters():
        owt = OWTStream(config['owt_tokens'], model.backbone.get_input_embeddings().num_embeddings, seed=config['seed'] + rank, tokenizer_identity=None if args.tiny else sha256_file(Path(config['backbone_path']) / 'tokenizer.json'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest(config, data, 'training_diagnostic' if diagnostic else 'training_run', diagnostic)
    manifest.update({'task': args.task, 'world_size': world, 'batch_size_per_rank': args.batch_size, 'configured_optimizer_steps': total, 'tokenizer': model.vq.provenance, 'adaptation': asdict(model.adapter_report), 'statistics_sha256': sha256_file(args.statistics)})
    manifest['execution_recipe'] = {'task': args.task, 'epochs': epochs, 'step_budget': total, 'batch_size': args.batch_size, 'reference_batch': args.reference_batch or config['reference_batch'], 'reference_length': args.reference_length or config['reference_length'], 'tiny': args.tiny, 'dataset_names': names, 'sample_limit': args.limit}
    if args.task == 'generation':
        manifest['execution_recipe']['fields'] = list(requested_fields)
        manifest['generation_recipe'] = {'fields': list(requested_fields), 'expanded_training_examples': len(training_data), 'epoch_unit': 'one pass over each independently queried field per window', 'prompt_bank': FIELD_PARAPHRASES, 'prefix_bank': FIELD_PREFIXES, 'paraphrase_policy': 'seeded stateless bank selection per sample and epoch', 'target_policy': config.get('target_policy', 'independent_waveform_or_annotation'), 'teacher_llm_refinement': False, 'smoke_template_targets': bool(getattr(args, 'smoke_template_targets', False))}
    if args.task == 'generation':
        manifest['generation_recipe']['target_files_sha256'] = {name: sha256_file(index.path) for (name, index) in getattr(data, 'targets', {}).items() if index.path.is_file()}
    manifest['resource_identity'] = resource_identity(config, model, data, args.statistics, args.tiny)
    manifest['parameter_counts'] = model.parameter_counts()
    manifest['stage_initialization'] = {'checkpoint_sha256': sha256_file(args.initialize_from) if args.initialize_from else None, 'shared_next_token_pretraining_loaded': bool(args.initialize_from), 'new_optimizer_and_task_head': bool(args.initialize_from)}
    if args.task == 'pretraining':
        manifest['pretraining_targets'] = {'source': 'deterministic numerical descriptor sentences or supplied independent targets', 'teacher_llm_refinement': False}
    if owt:
        manifest['owt_training'] = {**owt.identity, 'sha256': sha256_file(config['owt_tokens'])}
    (start_epoch, step, best, resume_batch) = (0, 0, -math.inf, 0)
    if args.resume:
        payload = load_adaptation(args.resume, model, optimizer, scheduler, expected_identity=manifest['resource_identity'], restore_rng=world == 1)
        for key in ('config', 'implementation_sha256', 'execution_recipe', 'generation_recipe'):
            if payload['manifest'].get(key) != manifest.get(key):
                raise ValueError('Exact resume requires the original code and training recipe: ' + key)
        (start_epoch, step) = (payload['epoch'], payload['step'])
        state = payload.get('runtime_state', {})
        (resume_batch, best) = (state.get('batches_completed_in_epoch', 0), state.get('best_validation', -math.inf))
        if state.get('world_size', world) != world:
            raise ValueError('Exact resume requires the original distributed world size')
        if payload['manifest']['configured_optimizer_steps'] != total:
            raise ValueError('Exact resume requires the original optimizer step budget and scheduler')
        rank_state = state.get('per_rank_states', [])[rank] if state.get('per_rank_states') else state
        if world > 1:
            if not state.get('per_rank_states'):
                raise ValueError("Distributed resume requires a checkpoint containing every rank's RNG and OWT state")
            torch.set_rng_state(rank_state['torch_rng'])
            random.setstate(rank_state['python_rng'])
            np.random.set_state(rank_state['numpy_rng'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state(rank_state['cuda_rng'], device)
        if owt and rank_state.get('owt_generator'):
            owt.generator.bit_generator.state = rank_state['owt_generator']
        manifest['stage_initialization'] = payload['manifest'].get('stage_initialization', manifest['stage_initialization'])
    if rank == 0:
        save_json(output / 'manifest.json', manifest)
    history = json.loads((output / 'history.json').read_text()) if args.resume and (output / 'history.json').exists() else []
    if step >= total:
        if rank == 0:
            save_json(output / 'completion.json', {'completed_optimizer_steps': step, 'completed_epochs': start_epoch, 'artifact_kind': manifest['artifact_kind'], 'diagnostic': diagnostic, 'full_budget_completed': step >= epochs * len(loader)})
        data.close()
        return
    for epoch in range(start_epoch, epochs):
        order_generator.manual_seed(config['seed'] + epoch)
        if args.task == 'generation':
            training_data.epoch = epoch
        if sampler:
            sampler.set_epoch(epoch)
        completed_batches = 0
        for (batch_index, raw) in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch:
                completed_batches += 1
                continue
            batch = move_batch(raw, device)
            (batch['prompts'], batch['targets']) = (batch['prompt'], batch['target'])
            result = online_step(model, optimizer, batch, text, owt, task=args.task, reference_k=config['reference_k'], reference_batch=args.reference_batch or config['reference_batch'], reference_length=args.reference_length or config['reference_length'], rho=config['rho'], lambda_cov=config['lambda_cov'], clip_norm=config['clip_norm'], max_target_tokens=config['max_target_tokens'])
            scheduler.step()
            step += 1
            result.update({'step': step, 'epoch': epoch, 'lr': scheduler.get_last_lr()[0]})
            history.append(result)
            completed_batches = batch_index + 1
            if rank == 0:
                print(json.dumps(result), flush=True)
                save_json(output / 'history.json', history)
            if args.max_steps and step >= args.max_steps or (args.stop_after_steps and step >= args.stop_after_steps):
                break
        whole_epoch = completed_batches == len(loader)
        next_epoch = epoch + 1 if whole_epoch else epoch
        runtime_state = {'batches_completed_in_epoch': 0 if whole_epoch else completed_batches, 'best_validation': best, 'owt_generator': owt.generator.bit_generator.state if owt else None, 'world_size': world}
        if world > 1:
            local_state = {'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None, 'owt_generator': owt.generator.bit_generator.state if owt else None}
            states = [None] * world
            torch.distributed.all_gather_object(states, local_state)
            runtime_state['per_rank_states'] = states
        if args.task == 'classification' and (not diagnostic):
            val = dataset(config, names, 'val', scaler=scaler)
            (metrics, _, _) = validation(model, val, device, args.batch_size)
            score = metrics['metrics']['balanced_accuracy']
            if score is None:
                raise ValueError('Validation has no valid balanced accuracy')
            if score > best:
                best = score
                runtime_state['best_validation'] = best
                if rank == 0:
                    save_checkpoint(output / 'best.pt', model, optimizer, scheduler, next_epoch, step, manifest, runtime_state)
                    save_json(output / 'best_validation.json', metrics)
            val.close()
        if rank == 0:
            runtime_state['best_validation'] = best
            save_checkpoint(output / 'last.pt', model, optimizer, scheduler, next_epoch, step, manifest, runtime_state)
        if args.max_steps and step >= args.max_steps or (args.stop_after_steps and step >= args.stop_after_steps):
            break
    if rank == 0:
        save_json(output / 'completion.json', {'completed_optimizer_steps': step, 'completed_epochs': next_epoch, 'artifact_kind': manifest['artifact_kind'], 'diagnostic': diagnostic, 'full_budget_completed': step >= epochs * len(loader)})
    data.close()

def command_evaluate(args, config):
    (_, _, device) = distributed_context()
    seed_everything(config['seed'])
    if len(args.datasets) != 1:
        raise ValueError('Classification evaluation requires one class vocabulary')
    name = args.datasets[0]
    scaler = DescriptorStandardizer.load(args.statistics)
    data = dataset(config, name, args.split, limit=args.limit, scaler=scaler, diagnostic=args.diagnostic)
    depth = config['classification_head_layers'].get(name, config['classification_head_layers']['default'])
    (model, _) = model_and_text(config, device, PAPER[name]['classes'], depth, args.tiny)
    load_adaptation(args.checkpoint, model, expected_identity=resource_identity(config, model, data, args.statistics, args.tiny))
    (metrics, probabilities, labels) = validation(model, data, device, args.batch_size)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'metrics.json', {'artifact_kind': 'classification_diagnostic' if args.diagnostic else 'classification_evaluation', 'diagnostic': args.diagnostic, 'checkpoint_sha256': sha256_file(args.checkpoint), 'dataset': name, 'split': args.split, 'metrics': metrics})
    with (output / 'predictions.jsonl').open('w') as stream:
        for (row, truth, p) in zip(data.records, labels, probabilities):
            stream.write(json.dumps({'sample_id': digest_json([name, row['sample_key']]), 'label': truth, 'probabilities': p.tolist()}) + '\n')
    print(json.dumps(metrics))
    data.close()

def command_generate(args, config):
    (_, _, device) = distributed_context()
    seed_everything(config['seed'])
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config['max_new_tokens']
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('Generation token budget must be a positive integer')
    scaler = DescriptorStandardizer.load(args.statistics)
    data = dataset(config, args.datasets, args.split, limit=args.limit, scaler=scaler, diagnostic=args.diagnostic)
    try:
        (model, text) = model_and_text(config, device, tiny=args.tiny)
        identity = resource_identity(config, model, data, args.statistics, args.tiny)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        manifest = payload.get('manifest', {})
        source = manifest.get('resource_identity', {})
        if manifest.get('task') != 'generation' or manifest.get('diagnostic') is not args.diagnostic:
            raise ValueError('Generation requires a generation-stage checkpoint with the same diagnostic status')
        for key in ('base', 'model_semantics', 'vq_sha256', 'statistics_sha256', 'language_training_stream_sha256'):
            if source.get(key) != identity.get(key):
                raise ValueError('Generation checkpoint resource identity differs: ' + key)
        source_datasets = source.get('dataset_manifest_hashes', {})
        if any((source_datasets.get(name) != fingerprint for (name, fingerprint) in identity['dataset_manifest_hashes'].items())):
            raise ValueError('Generation dataset manifests must match a subset of the training checkpoint')
        del payload
        load_adaptation(args.checkpoint, model)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('w') as stream:
            for raw in DataLoader(data, batch_size=args.batch_size, collate_fn=collate):
                batch = move_batch(raw, device)
                prompt = text([FIELD_PROMPTS[args.field] + ' Dataset: ' + name + '.' for name in batch['dataset_name']], padding=True, return_tensors='pt')
                with autocast_for(device):
                    generated = model.generate(**eeg_arguments(batch), prompt_ids=prompt['input_ids'].to(device), prompt_attention_mask=prompt['attention_mask'].to(device), max_new_tokens=max_new_tokens, eos_token_id=text.eos_token_id, information_condition=args.condition)
                for (i, tokens) in enumerate(generated):
                    stream.write(json.dumps({'sample_id': batch['sample_ids'][i], 'dataset': batch['dataset_name'][i], 'field': args.field, 'text': text.decode(tokens.cpu(), skip_special_tokens=True), 'generated_token_count': len(tokens), 'diagnostic': args.diagnostic, 'condition': args.condition}) + '\n')
        save_json(destination.with_suffix('.metadata.json'), {'artifact_kind': 'autoregressive_predictions', 'diagnostic': args.diagnostic, 'checkpoint_sha256': sha256_file(args.checkpoint), 'predictions_sha256': sha256_file(destination), 'resource_identity': identity, 'datasets': args.datasets, 'split': args.split, 'field': args.field, 'condition': args.condition, 'max_new_tokens': max_new_tokens, 'sample_count': len(data)})
        print('autoregressive_predictions', len(data))
    finally:
        data.close()

def command_build_grounding(args, config):
    from .grounding import build_grounding
    independent = copy.deepcopy(config)
    independent['targets_path'] = ''
    datasets = []
    try:
        for split in ('train', 'val', 'test'):
            datasets.append(dataset(independent, args.datasets, split, limit=args.limit, diagnostic=args.diagnostic))
        report = build_grounding(*datasets, args.output)
        print(json.dumps({'provenance': report['provenance'], 'datasets': {name: record['counts'] for (name, record) in report['conditions'].items()}}))
        return report
    finally:
        for current in datasets:
            current.close()

def command_build_field_parser(args, config):
    from .evaluation import build_field_parser
    result = build_field_parser(args.targets, args.output)
    print(json.dumps({'kind': result['kind'], 'training_target_count': result['training_target_count'], 'dataset_count': len(result['label_sets'])}))
    return result

def command_score_fields(args, config):
    from .evaluation import score_fields
    result = score_fields(args.predictions, args.targets, args.parser_config, predictions_metadata_path=args.predictions_metadata, sample_ids_path=args.sample_ids)
    save_json(args.output, result)
    print(json.dumps(result))
    return result

def command_score_generation(args, config):
    from .evaluation import score_summaries
    result = score_summaries(args.predictions, args.targets, args.direct_reference, args.withheld_reference, predictions_metadata_path=args.predictions_metadata, sample_ids_path=args.sample_ids)
    save_json(args.output, result)
    print(json.dumps(result))
    return result

def command_train_vq(args, config):
    require_paths(config, 'text_model')
    (rank, world, device) = distributed_context()
    if world != 1:
        raise ValueError('Tokenizer contrastive training currently requires one process; global SigLIP negatives are not approximated by rank-local negatives')
    seed_everything(config['seed'])
    data = dataset(config, args.datasets, 'train', limit=args.limit, diagnostic=args.diagnostic)
    from transformers import AutoTokenizer, AutoModel
    text_tokenizer = AutoTokenizer.from_pretrained(config['text_model'], local_files_only=True)
    text_model = AutoModel.from_pretrained(config['text_model'], local_files_only=True).to(device).eval().requires_grad_(False)
    text_dim = text_model.config.hidden_size
    cfg = EncoderConfig(n_layer=args.encoder_layers, n_embd=args.encoder_width, n_head=args.encoder_heads)
    model = PaperVQTokenizer(cfg, text_dim=text_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001, fused=device.type == 'cuda')
    loader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate, shuffle=True, drop_last=True)
    if not len(loader):
        raise ValueError('At least two paired windows are required for sigmoid contrastive alignment')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(loader))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest(config, data, 'tokenizer_training_diagnostic' if args.diagnostic else 'tokenizer_training', args.diagnostic)
    manifest.update({'encoder_config': asdict(cfg), 'optimizer': {'name': 'AdamW', 'lr': 0.0001, 'betas': [0.9, 0.999], 'weight_decay': 0.0001, 'clipping': False}, 'paired_text_source': 'deterministic_numerical_descriptor_sentences', 'teacher_llm_refinement': False, 'text_encoder_dimension': text_dim, 'siglip_negatives': 'whole local batch', 'domain_text': 'uniform frozen text embedding token samples'})
    save_json(output / 'manifest.json', manifest)
    total = args.epochs * len(loader)
    (history, step) = ([], 0)
    for epoch in range(args.epochs):
        for raw in loader:
            batch = move_batch(raw, device)
            text_inputs = text_tokenizer(batch['descriptor_text'], padding=True, truncation=True, max_length=512, return_tensors='pt')
            text_inputs = {key: value.to(device) for (key, value) in text_inputs.items()}
            with torch.no_grad():
                text_out = text_model(**text_inputs).last_hidden_state
                mask = text_inputs['attention_mask'].unsqueeze(-1)
                pooled = (text_out * mask).sum(1) / mask.sum(1).clamp_min(1)
                random_ids = torch.randint(0, text_model.config.vocab_size, (len(raw['labels']), 64), device=device)
                random_text_tokens = text_model.get_input_embeddings()(random_ids)
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(device):
                losses = model(batch['eeg'], batch['input_chans'], batch['input_times'], batch['input_mask'], pooled, random_text_tokens, progress=step / max(1, total - 1))
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite tokenizer loss')
            losses['loss'].backward()
            optimizer.step()
            scheduler.step()
            step += 1
            history.append({key: float(value.detach()) for (key, value) in losses.items() if key != 'codes'})
            if args.max_steps and step >= args.max_steps:
                break
        payload = {'format': 'ewen-paper-vq-v1', 'state_dict': model.state_dict(), 'encoder_config': asdict(cfg), 'n_embed': model.n_embed, 'code_dim': model.quantizer.dim, 'text_dim': text_dim, 'training_manifest': manifest, 'completed_steps': step}
        torch.save(payload, output / 'vq.pt')
        save_json(output / 'history.json', history)
        if args.max_steps and step >= args.max_steps:
            break
    print('tokenizer_training_steps', step, 'raw_huber+EMA_VQ+SigLIP+domain_reversal', flush=True)
    data.close()

def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)

def public_config(config):
    hidden = {'data_root', 'vq_checkpoint', 'backbone_path', 'variant_backbone_paths', 'variant_owt_tokens', 'variant_owt_eval_tokens', 'variant_owt_eval_provenance', 'pretraining_checkpoints', 'owt_tokens', 'owt_eval_tokens', 'owt_eval_provenance', 'text_model', 'targets_path', 'annotation_path', 'manifest_path'}
    hidden.add('method_pretraining_checkpoints')
    return {key: '<local-resource>' if key in hidden and value else value for (key, value) in config.items()}

def environment():
    import h5py
    import scipy
    import transformers
    result = {'python': sys.version.split()[0], 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__, 'h5py': h5py.__version__, 'scipy': scipy.__version__, 'cuda': torch.version.cuda, 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'cudnn_deterministic': torch.backends.cudnn.deterministic, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if torch.cuda.is_available():
        result['device'] = torch.cuda.get_device_name()
    return result

def read_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = getattr(args, 'config', None)
    if path:
        config.update(json.loads(Path(path).read_text()))
    return config

def parser():
    p = argparse.ArgumentParser(prog='ewen')
    p.add_argument('--config')
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('audit', 'prepare-data', 'fit-statistics', 'train', 'evaluate', 'generate', 'build-grounding'):
        sub = commands.add_parser(name)
        sub.add_argument('--datasets', nargs='+', choices=DATASET_NAMES, default=list(DATASET_NAMES) if name in {'audit', 'prepare-data'} else ['mental-arithmetic'])
        sub.add_argument('--output', required=True)
        if name in {'fit-statistics', 'train', 'evaluate', 'generate', 'build-grounding'}:
            sub.add_argument('--limit', type=int)
            sub.add_argument('--diagnostic', action='store_true')
        if name in {'train', 'evaluate', 'generate'}:
            sub.add_argument('--statistics', required=True)
            sub.add_argument('--batch-size', type=int, default=8)
            sub.add_argument('--tiny', action='store_true')
        if name == 'train':
            sub.add_argument('--task', choices=['classification', 'pretraining', 'generation'], default='classification')
            sub.add_argument('--fields', nargs='+', choices=list(FIELD_PROMPTS))
            sub.add_argument('--smoke-template-targets', action='store_true')
            sub.add_argument('--epochs', type=int)
            sub.add_argument('--max-steps', type=int)
            sub.add_argument('--stop-after-steps', type=int)
            sub.add_argument('--workers', type=int, default=0)
            sub.add_argument('--reference-batch', type=int)
            sub.add_argument('--reference-length', type=int)
            sub.add_argument('--resume')
            sub.add_argument('--initialize-from')
        if name == 'evaluate':
            sub.add_argument('--checkpoint', required=True)
            sub.add_argument('--split', choices=['val', 'test'], default='test')
        if name == 'generate':
            sub.add_argument('--checkpoint', required=True)
            sub.add_argument('--split', choices=['val', 'test'], default='test')
            sub.add_argument('--field', choices=list(FIELD_PROMPTS), default='summary')
            sub.add_argument('--condition', choices=['full', 'waveform_only', 'descriptor_only', 'prompt_only'], default='full')
            sub.add_argument('--max-new-tokens', type=int)
            sub.add_argument('--smoke-template-targets', action='store_true')
    sub = commands.add_parser('build-field-parser')
    sub.add_argument('--targets', nargs='+', required=True)
    sub.add_argument('--output', required=True)
    for name in ('score-fields', 'score-generation'):
        sub = commands.add_parser(name)
        sub.add_argument('--predictions', required=True)
        sub.add_argument('--targets', required=True)
        sub.add_argument('--predictions-metadata')
        sub.add_argument('--sample-ids')
        sub.add_argument('--output', required=True)
        if name == 'score-fields':
            sub.add_argument('--parser-config', required=True)
        else:
            sub.add_argument('--direct-reference', required=True)
            sub.add_argument('--withheld-reference', required=True)
    sub = commands.add_parser('train-vq')
    sub.add_argument('--datasets', nargs='+', choices=DATASET_NAMES, default=list(DATASET_NAMES))
    sub.add_argument('--output', required=True)
    sub.add_argument('--epochs', required=True, type=int)
    sub.add_argument('--max-steps', type=int)
    sub.add_argument('--limit', type=int)
    sub.add_argument('--batch-size', type=int, default=64)
    sub.add_argument('--encoder-layers', type=int, default=12)
    sub.add_argument('--encoder-width', type=int, default=768)
    sub.add_argument('--encoder-heads', type=int, default=12)
    sub.add_argument('--diagnostic', action='store_true')
    return p

def dispatch(args, config):
    commands = {'audit': command_audit, 'prepare-data': command_prepare, 'fit-statistics': command_statistics, 'train': command_train, 'evaluate': command_evaluate, 'train-vq': command_train_vq, 'generate': command_generate, 'build-grounding': command_build_grounding, 'build-field-parser': command_build_field_parser, 'score-fields': command_score_fields, 'score-generation': command_score_generation}
    if getattr(args, 'tiny', False) and (not getattr(args, 'diagnostic', False)):
        raise ValueError('TinyBackbone requires diagnostic execution')
    if getattr(args, 'smoke_template_targets', False) and (not getattr(args, 'diagnostic', False)):
        raise ValueError('Template descriptions require explicit diagnostic execution')
    return commands[args.command](args, config)

def run_command(argv, config):
    args = parser().parse_args(argv)
    settings = copy.deepcopy(DEFAULT_CONFIG)
    settings.update(copy.deepcopy(config))
    try:
        return dispatch(args, settings)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    args = parser().parse_args()
    dispatch(args, read_config(args))
if __name__ == '__main__':
    main()
