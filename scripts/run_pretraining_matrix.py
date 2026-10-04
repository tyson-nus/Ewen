import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NAMES = ["TUAB", "TUEV", "mental-arithmetic", "SHU-MI", "mumtaz", "BCICIV_2a", "Speech", "SEEDV"]
VARIANTS = ["S", "B", "L"]


def add_resources(parser):
    parser.add_argument("--config", default="")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--vq-checkpoint", default="")
    parser.add_argument("--manifest-path", default="")
    for variant in VARIANTS:
        suffix = variant.lower()
        parser.add_argument("--backbone-" + suffix, default="")
        parser.add_argument("--owt-tokens-" + suffix, default="")


def add_execution(parser):
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-length", type=int)
    parser.add_argument("--reference-batch", type=int)


def load_resources(args):
    from ewen_repro.cli import read_config
    if args.tiny and not args.diagnostic:
        raise ValueError("--tiny requires --diagnostic")
    if not args.diagnostic and any(getattr(args, name) is not None for name in ("epochs", "max_steps", "limit")):
        raise ValueError("--epochs, --max-steps and --limit require --diagnostic")
    for name in ("batch_size", "epochs", "max_steps", "limit", "reference_length", "reference_batch"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError("--" + name.replace("_", "-") + " must be positive")
    config = read_config(args)
    for key in ("data_root", "vq_checkpoint"):
        if getattr(args, key):
            config[key] = str(Path(getattr(args, key)).expanduser().resolve())
    if args.manifest_path:
        path = Path(args.manifest_path).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".json":
            mapping = json.loads(path.read_text())
            if not isinstance(mapping, dict):
                raise ValueError("Manifest path mapping must be a JSON object")
            config["manifest_path"] = {name: mapping_path(value, path.parent, "manifest_path." + name)
                                       for name, value in mapping.items()}
        else:
            config["manifest_path"] = str(path)
    for key, argument in (("variant_backbone_paths", "backbone"), ("variant_owt_tokens", "owt_tokens")):
        config[key] = dict(config.get(key) or {})
        for variant in VARIANTS:
            value = getattr(args, argument + "_" + variant.lower())
            if value:
                config[key][variant] = str(Path(value).expanduser().resolve())
    return config


def select_variant(config, variant, tiny=False, method=None):
    from ewen_repro.cli import require_paths
    result = copy.deepcopy(config)
    result["variant"] = variant
    if method is not None:
        result["adapter_mode"] = method
    for key, mapping in (("backbone_path", "variant_backbone_paths"), ("owt_tokens", "variant_owt_tokens")):
        value = result.get(mapping, {}).get(variant)
        if value:
            result[key] = value
        elif variant != config.get("variant", "S"):
            result[key] = ""
    keys = ["data_root", "vq_checkpoint"]
    if not tiny:
        keys.append("backbone_path")
    if result["adapter_mode"] == "structured":
        keys.append("owt_tokens")
    require_paths(result, *keys)
    return result


def run_command(argv, config):
    from ewen_repro.cli import run_command as execute
    return execute([str(value) for value in argv], copy.deepcopy(config))


def statistics_arguments(args, datasets, destination):
    values = ["fit-statistics", "--datasets", *datasets, "--output", destination]
    if args.diagnostic:
        values.append("--diagnostic")
    if args.limit is not None:
        values.extend(["--limit", args.limit])
    return values


def training_arguments(args):
    values = ["--batch-size", args.batch_size]
    for name in ("epochs", "max_steps", "limit", "reference_length", "reference_batch"):
        value = getattr(args, name)
        if value is not None:
            values.extend(["--" + name.replace("_", "-"), value])
    if args.diagnostic:
        values.append("--diagnostic")
    if args.tiny:
        values.append("--tiny")
    return values


def evaluation_arguments(args):
    values = ["--batch-size", args.batch_size]
    if args.diagnostic:
        values.append("--diagnostic")
    if args.limit is not None:
        values.extend(["--limit", args.limit])
    if args.tiny:
        values.append("--tiny")
    return values


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_output(args, config, selection):
    output = Path(args.output).expanduser().resolve()
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = {"config_sha256": fingerprint, "selection": selection,
              "execution": {name: getattr(args, name) for name in ("batch_size", "diagnostic", "tiny", "epochs", "max_steps", "limit", "reference_length", "reference_batch")}}
    path = output / "matrix.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Existing output belongs to a different matrix configuration; use another output directory")
    output.mkdir(parents=True, exist_ok=True)
    save_json(path, record)
    return output


def add_initialization(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretraining-checkpoints", default="")
    group.add_argument("--initialize-from", default="")


def mapping_path(value, base, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty path is required for " + field)
    return str((base / Path(value).expanduser()).resolve())


def validate_unique(args, *fields):
    for field in fields:
        values = getattr(args, field)
        if len(set(values)) != len(values):
            raise ValueError("--" + field.replace("_", "-") + " must contain unique values")


def load_initialization(args, config):
    if args.initialize_from:
        return {"single": str(Path(args.initialize_from).expanduser().resolve())}
    if args.pretraining_checkpoints:
        path = Path(args.pretraining_checkpoints).expanduser().resolve()
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Pretraining checkpoint mapping must be a JSON object")
        value = copy.deepcopy(value)
        for key in ("pretraining_checkpoints", "method_pretraining_checkpoints"):
            for variant, item in value.get(key, {}).items():
                if isinstance(item, dict):
                    value[key][variant] = {method: mapping_path(checkpoint, path.parent, key + "." + variant + "." + method)
                                           for method, checkpoint in item.items()}
                elif isinstance(item, str):
                    value[key][variant] = mapping_path(item, path.parent, key + "." + variant)
                else:
                    raise ValueError("Invalid checkpoint mapping for " + variant)
        if not any(key in value for key in ("pretraining_checkpoints", "method_pretraining_checkpoints")):
            value = {"pretraining_checkpoints": {variant: mapping_path(checkpoint, path.parent, "pretraining_checkpoints." + variant)
                                                 for variant, checkpoint in value.items()}}
        return value
    return {key: copy.deepcopy(config.get(key, {})) for key in ("pretraining_checkpoints", "method_pretraining_checkpoints")}


def stage_checkpoint(mapping, variant, method):
    value = mapping.get("single") or mapping.get("method_pretraining_checkpoints", {}).get(variant, {}).get(method)
    if not value and method == "structured":
        value = mapping.get("pretraining_checkpoints", {}).get(variant)
    if not value or not Path(value).is_file():
        raise ValueError("A matching pretraining checkpoint is required for " + variant + "/" + method)
    return str(Path(value).resolve())


def summarize(records, expected_seeds, diagnostic):
    from ewen_repro.metrics import summarize_runs
    groups = {}
    for row in records:
        key = (row["dataset"], row["variant"], json.dumps(row.get("recipe", {}), sort_keys=True))
        groups.setdefault(key, []).append(row)
    results = []
    for (name, variant, recipe), rows in groups.items():
        metrics = [row["metrics"]["metrics"] for row in rows]
        valid = all(value["validity"] == "valid" for value in metrics)
        scores = {key: summarize_runs([value["metrics"][key] for value in metrics]) for key in metrics[0]["metrics"]} if valid else {}
        results.append({"dataset": name, "variant": variant, "recipe": json.loads(recipe), "seeds": [row["seed"] for row in rows],
                        "run_count": len(rows), "expected_run_count": len(expected_seeds), "validity": "valid" if valid else "invalid",
                        "diagnostic": diagnostic, "full_run_count_completed": len(rows) == len(expected_seeds), "scores": scores})
    return {"diagnostic": diagnostic, "artifact_kind": "matrix_diagnostic_summary" if diagnostic else "computed_classification_run_summary", "groups": results}


def main():
    parser = argparse.ArgumentParser()
    add_resources(parser)
    add_execution(parser)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--datasets", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--methods", nargs="+", choices=["structured", "lora", "full"], default=["structured"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate_unique(args, "variants", "datasets", "methods")
    config = load_resources(args)
    if args.seed is not None:
        config["seed"] = args.seed
    for variant in args.variants:
        for method in args.methods:
            select_variant(config, variant, args.tiny, method)
    root = prepare_output(args, config, {"task": "pretraining", "datasets": args.datasets, "variants": args.variants, "methods": args.methods})
    statistics = root / "train_statistics.json"
    run_command(statistics_arguments(args, args.datasets, statistics), config)
    checkpoints = {"pretraining_checkpoints": {}, "method_pretraining_checkpoints": {}}
    for variant in args.variants:
        checkpoints["method_pretraining_checkpoints"][variant] = {}
        for method in args.methods:
            job = root / (variant + "_" + method) / "training"
            local = select_variant(config, variant, args.tiny, method)
            run_command(["train", "--task", "pretraining", "--datasets", *args.datasets,
                         "--statistics", statistics, "--output", job, *training_arguments(args)], local)
            checkpoint = job / "last.pt"
            if not checkpoint.is_file():
                raise ValueError("Pretraining did not produce a checkpoint")
            checkpoints["method_pretraining_checkpoints"][variant][method] = str(checkpoint)
            if method == "structured":
                checkpoints["pretraining_checkpoints"][variant] = str(checkpoint)
            save_json(root / "checkpoints.json", checkpoints)


if __name__ == "__main__":
    main()
