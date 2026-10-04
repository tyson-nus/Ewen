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
    parser.add_argument("--variant", choices=VARIANTS, default="B")
    parser.add_argument("--suite", choices=["covariate_geometry", "eeg_only", "mask"], default="covariate_geometry")
    parser.add_argument("--datasets", nargs="+", choices=NAMES, default=["SHU-MI", "mumtaz", "BCICIV_2a"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 37, 73, 101, 137])
    parser.add_argument("--mask-ratios", nargs="+", type=float, default=[0., .1, .2, .4, .6, .8])
    args = parser.parse_args()
    validate_unique(args, "datasets", "seeds", "mask_ratios")
    if any(not 0 <= ratio < 1 for ratio in args.mask_ratios):
        raise ValueError("Mask ratios must be in [0,1)")
    config = load_resources(args)
    initialization = load_initialization(args, config)
    local = select_variant(config, args.variant, args.tiny)
    stage = stage_checkpoint(initialization, args.variant, local["adapter_mode"])
    recipes = ([{"use_geometry": pe, "use_covariates": cov} for pe in (False, True) for cov in (False, True)]
               if args.suite == "covariate_geometry" else
               [{"use_geometry": False, "use_covariates": False}] if args.suite == "eeg_only" else
               [{"mask_ratio": ratio} for ratio in args.mask_ratios])
    root = prepare_output(args, config, {"task": "ablation", "suite": args.suite, "datasets": args.datasets,
        "variant": args.variant, "seeds": args.seeds, "recipes": recipes, "initialization": initialization})
    records = []
    for name in args.datasets:
        statistics = root / (name + ".train_statistics.json")
        run_command(statistics_arguments(args, [name], statistics), config)
        for recipe in recipes:
            recipe_name = "_".join(key + "=" + str(value) for key, value in recipe.items())
            for seed in args.seeds:
                local = select_variant(config, args.variant, args.tiny)
                local.update(recipe)
                local["seed"] = seed
                job = root / name / recipe_name / ("seed" + str(seed))
                run_command(["train", "--task", "classification", "--datasets", name, "--statistics", statistics,
                             "--initialize-from", stage, "--output", job / "training", *training_arguments(args)], local)
                checkpoint = job / "training" / ("last.pt" if args.diagnostic else "best.pt")
                if not checkpoint.is_file():
                    raise ValueError("Ablation did not produce its selected checkpoint")
                run_command(["evaluate", "--datasets", name, "--statistics", statistics, "--checkpoint", checkpoint,
                             "--output", job / "test", *evaluation_arguments(args)], local)
                records.append({"dataset": name, "seed": seed, "variant": args.variant, "recipe": recipe,
                                "diagnostic": args.diagnostic, "metrics": json.loads((job / "test" / "metrics.json").read_text())})
                save_json(root / "completed_runs.json", {"suite": args.suite, "mask_grid_is_reconstruction_choice": args.suite == "mask", "runs": records})
                save_json(root / "summary.json", summarize(records, args.seeds, args.diagnostic))


if __name__ == "__main__":
    main()
