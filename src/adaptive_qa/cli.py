import argparse
import json
from adaptive_qa.config import load_config, path_for
from adaptive_qa.baseline import prepare, run_baseline, fit_baseline_calibration


def main():
    parser = argparse.ArgumentParser(description="Adaptive QA research pipeline")
    parser.add_argument("command", choices=["prepare", "estimate", "baseline", "collect", "pilot-report"])
    parser.add_argument("--config", default="configs/fixture_core_check.yaml")
    parser.add_argument("--execute-model", action="store_true", help="Explicitly run the configured real model after reviewing estimate")
    args = parser.parse_args()
    c = load_config(args.config)
    from adaptive_qa.reproducibility import estimate, manifest
    if args.command == "estimate":
        print(json.dumps(estimate(c), indent=2))
        return
    out = path_for(c, c["output_dir"]) / c["experiment_id"]
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "pilot-report":
        from adaptive_qa.pilot import pilot_report
        print(json.dumps(pilot_report(c, out), indent=2))
        return
    if args.command == "prepare":
        qs, _ = prepare(c, out)
        print(f"Prepared {len(qs)} questions at {out}")
    elif args.command in {"baseline", "collect"}:
        if c["model"]["backend"] != "fixture" and not args.execute_model:
            raise ValueError("Review the estimate command, then pass --execute-model to execute local/API calls")
        manifest(c, out)
        if args.command == "collect":
            from adaptive_qa.outcomes import collect_outcomes
            from adaptive_qa.controller.utility import CostModel
            from adaptive_qa.controller.oracle import action_outcome_rows
            from adaptive_qa.io import write_jsonl
            rows, roots, _ = collect_outcomes(c, out)
            cm = CostModel(c["cost"], c["reward"])
            write_jsonl(out / "action_outcomes.jsonl", action_outcome_rows({r["node_id"]:r for r in rows}, roots, cm, c["lambdas"][0]))
            print(f"Saved {len(rows)} observed nodes. Dataset: {c['dataset']['name']}. Output: {out}")
            return
        rows = run_baseline(c, out)
        if c["calibration"]["enabled"]:
            fit_baseline_calibration(rows, out, c["thresholds"], c["evaluation"]["include_wait"])
        else:
            from adaptive_qa.pilot import pilot_report
            pilot_report(c, out)
        print(f"Saved {len(rows)} prefix predictions. Dataset: {c['dataset']['name']}. Output: {out}")


if __name__ == "__main__":
    main()
