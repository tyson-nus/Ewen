import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pretraining_matrix import (NAMES, VARIANTS, add_resources, add_execution, add_initialization,
    load_resources, load_initialization, select_variant, stage_checkpoint, prepare_output, statistics_arguments,
    training_arguments, evaluation_arguments, run_command, save_json, summarize, validate_unique)


def main():
    parser = argparse.ArgumentParser()
    add_resources(parser)
    add_execution(parser)
    add_initialization(parser)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--datasets", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101, 137])
    args = parser.parse_args()
    validate_unique(args, "variants", "datasets", "seeds")
    if args.initialize_from and len(args.variants) != 1:
        raise ValueError("--initialize-from requires exactly one variant")
    config = load_resources(args)
    initialization = load_initialization(args, config)
    for variant in args.variants:
        local = select_variant(config, variant, args.tiny)
        stage_checkpoint(initialization, variant, local["adapter_mode"])
    root = prepare_output(args, config, {"task": "classification", "datasets": args.datasets,
        "variants": args.variants, "seeds": args.seeds, "initialization": initialization})
    records = []
    for name in args.datasets:
        statistics = root / (name + ".train_statistics.json")
        run_command(statistics_arguments(args, [name], statistics), config)
        for variant in args.variants:
            for seed in args.seeds:
                local = select_variant(config, variant, args.tiny)
                local["seed"] = seed
                stage = stage_checkpoint(initialization, variant, local["adapter_mode"])
                job = root / (name + "_" + variant + "_seed" + str(seed))
                run_command(["train", "--task", "classification", "--datasets", name, "--statistics", statistics,
                             "--initialize-from", stage, "--output", job / "training", *training_arguments(args)], local)
                checkpoint = job / "training" / ("last.pt" if args.diagnostic else "best.pt")
                if not checkpoint.is_file():
                    raise ValueError("Classification did not produce its selected checkpoint")
                run_command(["evaluate", "--datasets", name, "--statistics", statistics, "--checkpoint", checkpoint,
                             "--output", job / "test", *evaluation_arguments(args)], local)
                records.append({"dataset": name, "variant": variant, "seed": seed, "diagnostic": args.diagnostic,
                                "metrics": json.loads((job / "test" / "metrics.json").read_text())})
                save_json(root / "completed_runs.json", records)
                save_json(root / "summary.json", summarize(records, args.seeds, args.diagnostic))


if __name__ == "__main__":
    main()
