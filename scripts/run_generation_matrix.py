import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pretraining_matrix import (VARIANTS, add_resources, add_execution, add_initialization,
    load_resources, load_initialization, select_variant, stage_checkpoint, prepare_output,
    statistics_arguments, training_arguments, evaluation_arguments, run_command, save_json,
    mapping_path, validate_unique)

DATASETS = ["mumtaz", "mental-arithmetic", "SHU-MI"]
FIELDS = ["label", "relation", "band", "summary"]
CONDITIONS = ["full", "waveform_only", "descriptor_only", "prompt_only"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resource_mapping(value, names, base, field):
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        path = Path(mapping_path(value, base, field))
        if path.suffix.lower() == ".json" and path.is_file():
            document = json.loads(path.read_text())
            if not isinstance(document, dict):
                raise ValueError(field + " mapping must be a JSON object")
            if document.get("kind") in {"direct_numeric", "withheld_states"}:
                if len(names) != 1:
                    raise ValueError(field + " requires a dataset mapping when multiple datasets are selected")
                value = {names[0]: str(path)}
            else:
                value = document
                base = path.parent
        elif len(names) == 1:
            value = {names[0]: str(path)}
        else:
            raise ValueError(field + " requires a dataset mapping when multiple datasets are selected")
    if not isinstance(value, dict):
        raise ValueError(field + " must be a JSON mapping or a single-dataset file")
    result = {name: mapping_path(value[name], base, field + "." + name) for name in names if name in value}
    for name, path in result.items():
        if not Path(path).is_file():
            raise ValueError(field + "." + name + " must identify an existing file")
    return result


def generation_resources(args, config):
    config_base = Path(args.config).expanduser().resolve().parent if args.config else Path.cwd()
    targets = resource_mapping(args.targets_path or config.get("targets_path"), args.datasets,
                               Path.cwd() if args.targets_path else config_base, "targets_path")
    missing = [name for name in args.datasets if name not in targets]
    if missing and not (args.diagnostic and args.smoke_template_targets):
        raise ValueError("Independent targets are required for " + ", ".join(missing))
    config["targets_path"] = targets
    references = {}
    for field, option in (("direct_reference", "direct_reference_path"), ("withheld_reference", "withheld_reference_path")):
        supplied = getattr(args, option) or config.get(option)
        values = resource_mapping(supplied, args.datasets,
                                  Path.cwd() if getattr(args, option) else config_base, option)
        for name, path in targets.items():
            candidate = Path(path).parent / (field + ".json")
            if name not in values and candidate.is_file():
                values[name] = str(candidate.resolve())
        if not args.diagnostic and any(name not in values for name in args.datasets):
            raise ValueError("Independent frozen " + field + " sidecars are required for every selected dataset")
        references[field] = values
    parser = args.field_parser or config.get("field_parser", "")
    if parser:
        parser = mapping_path(parser, Path.cwd() if args.field_parser else config_base, "field_parser")
        if not Path(parser).is_file():
            raise ValueError("field_parser must identify an existing frozen parser file")
    return targets, references, parser


def combine_fields(paths, destination, name, condition, diagnostic):
    if len(paths) != len(FIELDS):
        raise ValueError("All four generated fields are required")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    identity = None
    sample_ids = None
    total = 0
    with temporary.open("w") as output:
        for field, path in zip(FIELDS, paths):
            path = Path(path)
            metadata = json.loads(path.with_suffix(".metadata.json").read_text())
            if (metadata.get("split") != "test" or metadata.get("datasets") != [name]
                    or metadata.get("field") != field or metadata.get("condition") != condition
                    or metadata.get("diagnostic") is not diagnostic or metadata.get("predictions_sha256") != sha(path)):
                raise ValueError("Generated field metadata must identify its exact test dataset, field and condition")
            comparable = {key: value for key, value in metadata.items()
                          if key not in {"field", "predictions_sha256", "sample_count"}}
            if identity is not None and comparable != identity:
                raise ValueError("Four-field outputs must share their checkpoint and generation recipe")
            identity = comparable
            current = set()
            with path.open() as source:
                for line in source:
                    row = json.loads(line)
                    sid = row.get("sample_id")
                    if (not isinstance(sid, str) or not sid or sid in current or row.get("dataset") != name
                            or row.get("field") != field or row.get("condition") != condition
                            or row.get("diagnostic") is not diagnostic or row.get("split", "test") != "test"):
                        raise ValueError("Generated fields must contain distinct matching test sample IDs")
                    current.add(sid)
                    output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    total += 1
            if not current or len(current) != metadata.get("sample_count"):
                raise ValueError("Generated field sample counts must match metadata")
            if sample_ids is not None and current != sample_ids:
                raise ValueError("All four task outputs must cover identical test windows")
            sample_ids = current
    temporary.replace(destination)
    save_json(destination.with_suffix(".metadata.json"), {**identity, "fields": FIELDS,
        "predictions_sha256": sha(destination), "sample_count": total})
    return sorted(sample_ids)


def skipped(path, diagnostic, reason):
    value = {"artifact_kind": "generation_scoring_status", "diagnostic": diagnostic,
             "scoring_status": "not_scored", "reason": reason}
    save_json(path, value)
    return value


def metric_values(field_result, summary_result):
    values, invalid = {}, []
    sources = {"field": field_result.get("field_accuracy")}
    sources.update({name: summary_result.get(name)
                    for name in ("summary_rouge_l", "direct_f1", "withheld_f1")})
    for category, value in sources.items():
        if not isinstance(value, dict) or value.get("validity") != "valid":
            invalid.append(category)
            continue
        for name, score in value["metrics"].items():
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                values[category + "." + name] = score
    return values, invalid


def summarize(records, expected_seeds, diagnostic):
    from ewen_repro.metrics import summarize_runs
    groups = {}
    for row in records:
        key = (row["dataset"], row["variant"], row["method"], row["condition"])
        groups.setdefault(key, []).append(row)
    result = []
    for (name, variant, method, condition), rows in groups.items():
        scores = [metric_values(row["field_metrics"], row["summary_metrics"])[0] for row in rows]
        shared = set.intersection(*(set(score) for score in scores))
        invalid = sorted({reason for row in rows for reason in metric_values(row["field_metrics"], row["summary_metrics"])[1]})
        result.append({"dataset": name, "variant": variant, "method": method, "condition": condition,
            "seeds": [row["seed"] for row in rows], "run_count": len(rows), "expected_run_count": len(expected_seeds),
            "full_run_count_completed": len(rows) == len(expected_seeds), "unscored_or_invalid_categories": invalid,
            "scores": {key: summarize_runs([score[key] for score in scores]) for key in sorted(shared)}})
    return {"artifact_kind": "generation_diagnostic_summary" if diagnostic else "computed_generation_run_summary",
            "diagnostic": diagnostic, "groups": result}


def main():
    parser = argparse.ArgumentParser()
    add_resources(parser)
    add_execution(parser)
    add_initialization(parser)
    parser.add_argument("--targets-path", default="")
    parser.add_argument("--direct-reference-path", default="")
    parser.add_argument("--withheld-reference-path", default="")
    parser.add_argument("--field-parser", default="")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--smoke-template-targets", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--methods", nargs="+", choices=["structured", "lora", "full"], default=["structured"])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73])
    args = parser.parse_args()
    validate_unique(args, "datasets", "variants", "methods", "conditions", "seeds")
    if (args.max_new_tokens is not None or args.smoke_template_targets) and not args.diagnostic:
        raise ValueError("--max-new-tokens and --smoke-template-targets require --diagnostic")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.initialize_from and (len(args.variants) != 1 or len(args.methods) != 1):
        raise ValueError("--initialize-from requires one variant and one adaptation method")
    config = load_resources(args)
    targets, references, field_parser = generation_resources(args, config)
    training_fields = FIELDS if len(targets) == len(args.datasets) else ["summary"]
    initialization = load_initialization(args, config)
    stages = {}
    for variant in args.variants:
        for method in args.methods:
            select_variant(config, variant, args.tiny, method)
            stages[variant + "/" + method] = stage_checkpoint(initialization, variant, method)
    resource_hashes = {"targets": {name: sha(path) for name, path in targets.items()},
        **{key: {name: sha(path) for name, path in mapping.items()} for key, mapping in references.items()},
        "field_parser": sha(field_parser) if field_parser else None,
        "stage_initialization": {key: sha(path) for key, path in stages.items()}}
    output = prepare_output(args, config, {"task": "generation", "datasets": args.datasets,
        "variants": args.variants, "methods": args.methods, "conditions": args.conditions, "seeds": args.seeds,
        "initialization": initialization, "resource_hashes": resource_hashes,
        "max_new_tokens": args.max_new_tokens, "smoke_template_targets": args.smoke_template_targets,
        "training_fields": training_fields})
    statistics = output / "joint.train_statistics.json"
    run_command(statistics_arguments(args, args.datasets, statistics), config)
    if not field_parser and targets:
        field_parser = output / "frozen_field_parser.json"
        run_command(["build-field-parser", "--targets", *dict.fromkeys(targets.values()), "--output", field_parser], config)
    records = []
    for variant in args.variants:
        for method in args.methods:
            for seed in args.seeds:
                local = select_variant(config, variant, args.tiny, method)
                local["seed"] = seed
                job = output / (variant + "_" + method + "_seed" + str(seed))
                template = ["--smoke-template-targets"] if args.smoke_template_targets else []
                run_command(["train", "--task", "generation", "--datasets", *args.datasets, "--fields", *training_fields,
                    "--statistics", statistics, "--initialize-from", stages[variant + "/" + method],
                    "--output", job / "training", *training_arguments(args), *template], local)
                checkpoint = job / "training" / "last.pt"
                if not checkpoint.is_file():
                    raise ValueError("Generation training did not produce last.pt")
                for name in args.datasets:
                    for condition in args.conditions:
                        folder = job / "test" / name / condition
                        field_paths = []
                        for field in FIELDS:
                            path = folder / (field + ".jsonl")
                            token_limit = ["--max-new-tokens", args.max_new_tokens] if args.max_new_tokens is not None else []
                            run_command(["generate", "--datasets", name, "--split", "test", "--statistics", statistics,
                                "--checkpoint", checkpoint, "--field", field, "--condition", condition, "--output", path,
                                *evaluation_arguments(args), *token_limit, *template], local)
                            field_paths.append(path)
                        combined = folder / "all_fields.jsonl"
                        sample_ids = combine_fields(field_paths, combined, name, condition, args.diagnostic)
                        subset = []
                        if args.diagnostic:
                            sample_path = folder / "test_sample_ids.json"
                            save_json(sample_path, sample_ids)
                            subset = ["--sample-ids", sample_path]
                        field_output = folder / "field_metrics.json"
                        if name in targets and field_parser:
                            run_command(["score-fields", "--predictions", combined, "--targets", targets[name],
                                "--parser-config", field_parser, "--output", field_output, *subset], local)
                            field_metrics = json.loads(field_output.read_text())
                        else:
                            field_metrics = skipped(field_output, args.diagnostic, "Independent field targets or a frozen parser were not supplied")
                        summary_output = folder / "summary_metrics.json"
                        if name in targets and all(name in references[key] for key in references):
                            run_command(["score-generation", "--predictions", folder / "summary.jsonl", "--targets", targets[name],
                                "--direct-reference", references["direct_reference"][name],
                                "--withheld-reference", references["withheld_reference"][name], "--output", summary_output, *subset], local)
                            summary_metrics = json.loads(summary_output.read_text())
                        else:
                            summary_metrics = skipped(summary_output, args.diagnostic, "Independent summary targets and frozen direct/withheld sidecars were not supplied")
                        records.append({"dataset": name, "variant": variant, "method": method, "seed": seed,
                            "condition": condition, "diagnostic": args.diagnostic, "sample_count": len(sample_ids),
                            "checkpoint_sha256": sha(checkpoint), "field_metrics": field_metrics, "summary_metrics": summary_metrics})
                        save_json(output / "completed_runs.json", records)
                        save_json(output / "summary.json", summarize(records, args.seeds, args.diagnostic))


if __name__ == "__main__":
    main()
