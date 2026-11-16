"""Command-line interface:  rofoul <command> ...

    rofoul predict  plant.csv  -o out/        # normalized data, forecast, days-to-clean
    rofoul validate plant.csv                 # data-quality report only
    rofoul normalize plant.csv -o norm.csv    # normalized daily series only
    rofoul train    plant.csv  --out models/v2
    rofoul backtest plant.csv  --out reports/
    rofoul synth    --out data/synthetic      # regenerate the synthetic dataset
    rofoul serve                              # REST API on :8000
    rofoul info                               # metadata of the configured model

Exit codes: 0 ok, 2 input validation failed, 1 other error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .config import load_config, resolve_model_path
from .validation import InputValidationError, load_and_validate


def _cfg(args):
    over = {}
    if getattr(args, "model", None):
        over = {"model": {"path": args.model}}
    return load_config(args.config, overrides=over)


def cmd_validate(args) -> int:
    cfg = _cfg(args)
    try:
        _, report = load_and_validate(args.input, cfg)
    except InputValidationError as exc:
        report = exc.report
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.format())
        for tid, s in report.info.get("trains_summary", {}).items():
            print(f"  train {tid}: {s['start']} -> {s['end']}, {s['days_with_data']} days, "
                  f"sampling {s['sampling_interval_hours']} h, {s['n_gaps']} gaps")
    return 0 if report.ok else 2


def cmd_normalize(args) -> int:
    from .pipeline import DAILY_OUTPUT_COLS, prepare_daily
    cfg = _cfg(args)
    daily, report, refs = prepare_daily(args.input, cfg)
    out = daily[[c for c in DAILY_OUTPUT_COLS if c in daily.columns]]
    out.to_csv(args.output, index=False, float_format="%.4f")
    print(report.format())
    print(f"Wrote {len(out)} daily rows to {args.output}")
    return 0


def cmd_predict(args) -> int:
    from .pipeline import run
    cfg = _cfg(args)
    res = run(args.input, cfg)
    if args.json:
        print(json.dumps(res.to_dict(include_daily=False), indent=2, default=str))
        return 0
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    res.daily.to_csv(outdir / "normalized_daily.csv", index=False, float_format="%.4f")
    res.forecast.to_csv(outdir / "forecast.csv", index=False, float_format="%.4f")
    (outdir / "summary.json").write_text(json.dumps(
        {"summary": res.summary, "validation": res.report.to_dict(), "model": res.model_info,
         "references": res.references}, indent=2, default=str))
    print(res.report.format())
    print()
    print(f"{'train':<8}{'last day':<12}{'NPF decl %':>11}{'DP rise %':>10}{'days->clean':>13}  {'range':<14}{'alert':<7} message")
    for s in res.summary:
        if s.get("status") != "ok":
            print(f"{s['train_id']:<8}{'-':<12}{'':>11}{'':>10}{'':>13}  {'':<14}{'n/a':<7} {s.get('message', '')}")
            continue
        d = s["days_to_clean"]
        fmt = lambda v: "none" if v is None else f"{v:.0f}"
        rng = f"{fmt(d['earliest_p90_fouling'])}-{fmt(d['latest_p10_fouling'])}"
        cur = s["current"]
        print(f"{s['train_id']:<8}{s['last_data_date']:<12}{cur['npf_decline_pct'] or 0:>11.1f}"
              f"{cur['dp_rise_pct'] or 0:>10.1f}{fmt(d['estimate']):>13}  {rng:<14}{s['alert']:<7} {s['message']}")
        for w in s["warnings"]:
            print(f"{'':8}! {w}")
    print(f"\nOutputs written to {outdir.resolve()}/ (normalized_daily.csv, forecast.csv, summary.json)")
    return 0


def cmd_train(args) -> int:
    from .evaluate import train_model
    from .features import build_training_table
    from .pipeline import prepare_daily
    cfg = _cfg(args)
    if args.no_calibrate:
        cfg["training"]["calibrate"] = False
    daily, report, _ = prepare_daily(args.input, cfg)
    print(report.format())
    table = build_training_table(daily, cfg)
    print(f"Training on {table['train_id'].nunique()} train(s), {table['date'].nunique()} origin days, "
          f"{int(table['target_npf'].notna().sum())} labelled rows ...")
    model = train_model(table, cfg)
    meta = {
        "version": args.version or Path(args.out).name,
        "training_data": {"file": str(Path(args.input).name), "start": str(daily["date"].min().date()),
                          "end": str(daily["date"].max().date()), "trains": sorted(daily["train_id"].unique()),
                          "rows": int(len(table))},
        "config_snapshot": {k: cfg[k] for k in ("normalization", "cycles", "thresholds", "forecast", "training")},
    }
    path = model.save(args.out, meta)
    print(f"Model saved to {path}/ (decision quantile {model.decision_quantile}).")
    print(f"Activate it with  model.path: {args.out}  in your config, or  ROFOUL_MODEL_PATH={args.out}")
    return 0


def cmd_backtest(args) -> int:
    from .evaluate import backtest
    from .pipeline import prepare_daily
    cfg = _cfg(args)
    daily, report, _ = prepare_daily(args.input, cfg)
    res = backtest(daily, cfg, holdout_train=args.holdout_train)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"_holdout_{args.holdout_train}" if args.holdout_train else ""
    res["rows"].to_csv(outdir / f"backtest_rows{tag}.csv.gz", index=False, float_format="%.4f")
    res["days"].to_csv(outdir / f"backtest_days{tag}.csv", index=False, float_format="%.3f")
    (outdir / f"backtest_metrics{tag}.json").write_text(json.dumps(
        {"folds": res["folds"], "holdout_train": args.holdout_train, **res["metrics"]}, indent=2, default=str))
    print(json.dumps(res["metrics"], indent=2, default=str))
    return 0


def cmd_synth(args) -> int:
    from .synthetic import simulate, truth_with_indicators
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    data, truth = simulate(start=args.start, days=args.days, trains=tuple(args.trains), seed=args.seed,
                           include_cip_column=not args.no_cip_column)
    data.to_csv(outdir / "plant_data.csv", index=False)
    truth_with_indicators(truth).to_csv(outdir / "ground_truth_daily.csv", index=False, float_format="%.5f")
    print(f"Wrote {len(data)} rows to {outdir / 'plant_data.csv'} and ground truth to "
          f"{outdir / 'ground_truth_daily.csv'}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app
    cfg = _cfg(args)
    host = args.host or cfg["api"]["host"]
    port = int(args.port or cfg["api"]["port"])
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
    return 0


def cmd_info(args) -> int:
    cfg = _cfg(args)
    path = resolve_model_path(cfg)
    meta = path / "metadata.json"
    if not meta.exists():
        print(f"No model at {path}", file=sys.stderr)
        return 1
    print(f"Model directory: {path}")
    print(meta.read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rofoul", description="RO membrane fouling early-warning system")
    p.add_argument("--version", action="version", version=f"rofoul {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_input=True):
        if needs_input:
            sp.add_argument("input", help="plant operating data CSV")
        sp.add_argument("-c", "--config", help="YAML config (deep-merged over the defaults)")
        sp.add_argument("--model", help="model directory (overrides model.path)")
        return sp

    sp = common(sub.add_parser("validate", help="check an input CSV and report problems"))
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_validate)

    sp = common(sub.add_parser("normalize", help="write normalized daily data"))
    sp.add_argument("-o", "--output", default="normalized_daily.csv")
    sp.set_defaults(func=cmd_normalize)

    sp = common(sub.add_parser("predict", help="normalize, forecast fouling and estimate days to cleaning"))
    sp.add_argument("-o", "--output", default="rofoul_output", help="output directory")
    sp.add_argument("--json", action="store_true", help="print JSON to stdout instead of writing files")
    sp.set_defaults(func=cmd_predict)

    sp = common(sub.add_parser("train", help="train a new model version"))
    sp.add_argument("--out", required=True, help="output model directory, e.g. models/v2")
    sp.add_argument("--version", dest="version", help="version label stored in metadata")
    sp.add_argument("--no-calibrate", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = common(sub.add_parser("backtest", help="rolling-origin temporal validation vs baselines"))
    sp.add_argument("--out", default="reports")
    sp.add_argument("--holdout-train", help="exclude this train from training and test only on it")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("synth", help="generate the physics-based synthetic dataset")
    sp.add_argument("--out", default="data/synthetic")
    sp.add_argument("--start", default="2022-01-01")
    sp.add_argument("--days", type=int, default=1096)
    sp.add_argument("--trains", nargs="+", default=["A", "B", "C", "D"])
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--no-cip-column", action="store_true")
    sp.set_defaults(func=cmd_synth)

    sp = common(sub.add_parser("serve", help="start the REST API"), needs_input=False)
    sp.add_argument("--host")
    sp.add_argument("--port", type=int)
    sp.set_defaults(func=cmd_serve)

    sp = common(sub.add_parser("info", help="show the configured model's metadata"), needs_input=False)
    sp.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except InputValidationError as exc:
        print(exc.report.format(), file=sys.stderr)
        print("\nInput validation failed - see messages above.", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
