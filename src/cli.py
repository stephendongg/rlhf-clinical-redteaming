"""CLI entry point for rlhf-clinical-redteaming.

Subcommands: baseline, dpo, ppo.

Universal flags (apply to all subcommands):
    --config PATH        YAML config (required)
    --output-dir PATH    where to write per-run artifacts (default: results/runs)
    --seed INT           RNG seed (overrides config if present)
    --run-name STR       human-readable label; UUID is always also generated
    --allow-dirty        permit running on a dirty git tree
    --gcs-bucket STR     gs://<bucket> root for sync; if unset, no sync
    --no-gcs             disable GCS sync entirely
    --override KEY=VAL   repeated; dotted-path override into the resolved config
    --use-test           evaluate on test split (default: dev)

Method-specific flags live under their subcommand (`--beta` for dpo,
`--kl-coef` / `--lr` for ppo, etc.). Anything passed via --override
takes precedence over both the YAML and the typed flags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config as cfg
from . import env_capture
from . import setup_seed
from .results import RunRecord, ResultsLogger

log = logging.getLogger("redteam_rlhf.cli")


def _add_universal_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    p.add_argument("--output-dir", type=Path, default=Path("results/runs"),
                   help="Local dir for per-run artifacts")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (overrides config.seed)")
    p.add_argument("--run-name", type=str, default=None,
                   help="Human-readable run label")
    p.add_argument("--allow-dirty", action="store_true",
                   help="Permit running with uncommitted git changes")
    p.add_argument("--gcs-bucket", type=str, default=None,
                   help="gs://<bucket> root for sync (e.g. gs://results_043026)")
    p.add_argument("--no-gcs", action="store_true", help="Disable GCS sync")
    p.add_argument("--override", action="append", default=[],
                   help="Override config: --override path.to.key=value (repeatable)")
    p.add_argument("--use-test", action="store_true",
                   help="Evaluate on test split (default: dev)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redteam-rlhf",
        description="RLHF clinical red-teaming: baseline / DPO / PPO via CLI.",
    )
    sub = parser.add_subparsers(dest="method", required=True)

    # ── baseline ────────────────────────────────────────────────────────────
    p_base = sub.add_parser("baseline", help="Run untuned-attacker evaluation only.")
    _add_universal_flags(p_base)

    # ── dpo ─────────────────────────────────────────────────────────────────
    p_dpo = sub.add_parser("dpo", help="Iterative DPO training + eval.")
    _add_universal_flags(p_dpo)
    p_dpo.add_argument("--beta", type=float, default=None, help="DPO beta")
    p_dpo.add_argument("--lr", type=float, default=None, help="AdamW lr")
    p_dpo.add_argument("--n-outer", type=int, default=None,
                       help="Outer iterative loops")
    p_dpo.add_argument("--n-per-seed", type=int, default=None,
                       help="Rollouts per seed")
    p_dpo.add_argument("--n-epochs", type=int, default=None,
                       help="Inner epochs over cached pairs")

    # ── ppo ─────────────────────────────────────────────────────────────────
    p_ppo = sub.add_parser("ppo", help="PPO training + eval (TRL).")
    _add_universal_flags(p_ppo)
    p_ppo.add_argument("--lr", type=float, default=None, help="PPO lr")
    p_ppo.add_argument("--n-train-steps", type=int, default=None)
    p_ppo.add_argument("--batch-size", type=int, default=None)
    p_ppo.add_argument("--target-kl", type=float, default=None,
                       help="PPO target KL")
    p_ppo.add_argument("--kl-coef", type=float, default=None,
                       help="(alias for --target-kl in this codebase)")

    return parser


def _typed_flag_overrides(method: str, args: argparse.Namespace) -> dict:
    """Map subcommand-level typed flags into the dotted-path override schema."""
    out: dict[str, object] = {}
    if method == "dpo":
        for flag, key in [("beta", "dpo.beta"), ("lr", "dpo.lr"),
                          ("n_outer", "dpo.n_outer"),
                          ("n_per_seed", "dpo.n_per_seed"),
                          ("n_epochs", "dpo.n_epochs")]:
            v = getattr(args, flag)
            if v is not None:
                out[key] = v
    elif method == "ppo":
        for flag, key in [("lr", "ppo.lr"),
                          ("n_train_steps", "ppo.n_train_steps"),
                          ("batch_size", "ppo.batch_size"),
                          ("target_kl", "ppo.target_kl")]:
            v = getattr(args, flag)
            if v is not None:
                out[key] = v
        if args.kl_coef is not None:
            out["ppo.target_kl"] = args.kl_coef
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Load + override config.
    resolved = cfg.load_config(args.config)
    resolved = cfg.apply_overrides(resolved, _typed_flag_overrides(args.method, args))
    resolved = cfg.apply_overrides(resolved, cfg.parse_kv_overrides(args.override))
    resolved["use_test"] = bool(args.use_test or resolved.get("use_test", False))
    if args.seed is not None:
        resolved["seed"] = args.seed
    resolved.setdefault("seed", 42)

    if resolved.get("method", args.method) != args.method:
        log.warning("config method=%r but subcommand=%r; using subcommand.",
                    resolved.get("method"), args.method)
    resolved["method"] = args.method

    # 2. Capture env + check git.
    env = env_capture.capture(allow_dirty=args.allow_dirty)
    log.info("Git SHA: %s%s", env["git_sha"], " (dirty)" if env["git_dirty"] else "")
    log.info("GPU: %s", env.get("gpu_name", "n/a"))

    # 3. Seed everything.
    setup_seed.seed_everything(resolved["seed"])
    log.info("Seeded RNGs with seed=%d", resolved["seed"])

    # 4. Make a results logger / RunRecord.
    record = RunRecord.new(
        method=args.method,
        run_name=args.run_name,
        config=resolved,
        env=env,
    )
    log.info("Run ID: %s | name=%s", record.run_id, record.run_name)

    logger = ResultsLogger(
        record=record,
        output_dir=args.output_dir,
        gcs_bucket=None if args.no_gcs else args.gcs_bucket,
    )
    logger.write_initial()

    # 5. Dispatch to the method.
    try:
        if args.method == "baseline":
            from .methods import baseline
            metrics = baseline.run(resolved, logger)
        elif args.method == "dpo":
            from .methods import dpo as dpo_method
            metrics = dpo_method.run(resolved, logger)
        elif args.method == "ppo":
            from .methods import ppo as ppo_method
            metrics = ppo_method.run(resolved, logger)
        else:  # pragma: no cover
            raise ValueError(f"Unknown method: {args.method}")
    except Exception as e:
        log.exception("Run failed: %s", e)
        logger.finalize(status="failed", final_metrics={"error": repr(e)})
        return 1

    logger.finalize(status="ok", final_metrics=metrics)
    log.info("Done. Final metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
