"""Command-line entrypoint for end-to-end planner runs."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from ._shared.bootstrap import find_repo_root, list_traced_floorplan_names
from ._shared.config import PlannerConfig
from .k_runner import run_planner_batch, run_planner_batches_for_floorplans


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for explicit floor-plan and `K` selection runs."""

    parser = argparse.ArgumentParser(
        prog="python -m src.planner.runner",
        description=(
            "Run the end-to-end CCTV planner with explicit floor-plan and k-value "
            "selection."
        ),
    )

    floorplan_group = parser.add_mutually_exclusive_group(required=True)
    floorplan_group.add_argument(
        "--floorplan",
        action="append",
        dest="floorplan_names",
        metavar="NAME",
        help="Run one named floor plan. Repeat the flag to run more than one named plan.",
    )
    floorplan_group.add_argument(
        "--all-floorplans",
        action="store_true",
        help="Run every traced floor plan with matching PNG and JSON metadata.",
    )

    k_group = parser.add_mutually_exclusive_group(required=True)
    k_group.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        metavar="K",
        help="Run only the explicitly listed k values.",
    )
    k_group.add_argument(
        "--first-k-values",
        type=int,
        metavar="N",
        help="Run only the first N configured k values from PlannerConfig.k_values.",
    )
    k_group.add_argument(
        "--all-k-values",
        action="store_true",
        help="Run the full configured PlannerConfig.k_values tuple.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fresh recomputation instead of reusing valid cached artifacts.",
    )
    parser.add_argument(
        "--status-log",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional JSONL path that records one completed planner batch per line.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Optional floorplan-level worker count. When omitted, multi-floorplan runs "
            "use up to one worker per floorplan."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and run the requested planner batches."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)

    repo_root = find_repo_root()
    available_floorplan_names = list_traced_floorplan_names(repo_root=repo_root)
    floorplan_names = _resolve_requested_floorplan_names(
        parser,
        args,
        available_floorplan_names=available_floorplan_names,
    )

    base_config = PlannerConfig()
    requested_k_values = _resolve_requested_k_values(parser, args, base_config=base_config)

    progress_writer = sys.stderr
    if len(floorplan_names) == 1:
        run_planner_batch(
            replace(base_config, floorplan_name=floorplan_names[0]),
            k_values=requested_k_values,
            force=bool(args.force),
            status_log_path=args.status_log,
            progress_writer=progress_writer,
        )
        return 0

    run_planner_batches_for_floorplans(
        floorplan_names,
        base_config=base_config,
        k_values=requested_k_values,
        force=bool(args.force),
        status_log_path=args.status_log,
        progress_writer=progress_writer,
        max_workers=args.workers,
    )
    return 0


def _resolve_requested_floorplan_names(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    available_floorplan_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate and resolve the explicit floor-plan selection for one run."""

    if args.all_floorplans:
        if not available_floorplan_names:
            parser.error("No traced floor plans were found under static/floor-plan/traced.")
        return available_floorplan_names

    requested_floorplans = tuple(dict.fromkeys(args.floorplan_names or []))
    if not requested_floorplans:
        parser.error("Choose at least one --floorplan or pass --all-floorplans.")

    unknown_floorplans = sorted(
        floorplan_name
        for floorplan_name in requested_floorplans
        if floorplan_name not in available_floorplan_names
    )
    if unknown_floorplans:
        parser.error(
            "Unknown floorplan name(s): "
            + ", ".join(unknown_floorplans)
            + ". Available: "
            + ", ".join(available_floorplan_names)
        )
    return requested_floorplans


def _resolve_requested_k_values(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    base_config: PlannerConfig,
) -> tuple[int, ...]:
    """Validate and resolve the explicit `K` selection for one run."""

    if args.all_k_values:
        return tuple(base_config.k_values)

    if args.k_values is not None:
        requested_k_values = tuple(dict.fromkeys(int(k) for k in args.k_values))
        if not requested_k_values:
            parser.error("Pass at least one integer after --k-values.")
        if any(k <= 0 for k in requested_k_values):
            parser.error("Every explicit k value must be positive.")
        return requested_k_values

    first_k_values = int(args.first_k_values)
    if first_k_values <= 0:
        parser.error("--first-k-values must be positive.")
    if first_k_values > len(base_config.k_values):
        parser.error(
            "--first-k-values exceeds the configured k-value count "
            f"({len(base_config.k_values)})."
        )
    return tuple(base_config.k_values[:first_k_values])


if __name__ == "__main__":
    raise SystemExit(main())
