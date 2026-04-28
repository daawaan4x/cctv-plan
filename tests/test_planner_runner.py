"""Tests for the explicit CLI planner runner interface."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import ANY, patch

from src.planner._shared.config import PlannerConfig
from src.planner.k_runner import _PrefixedWriter, run_planner_batches_for_floorplans
from src.planner import runner


class PlannerRunnerTests(unittest.TestCase):
    """Verify the CLI selection rules and top-level dispatch behavior."""

    def test_main_requires_explicit_k_selection(self) -> None:
        with self.assertRaises(SystemExit):
            runner.main(["--floorplan", "ground-back"])

    @patch("src.planner.runner.run_planner_batch")
    @patch("src.planner.runner.list_traced_floorplan_names")
    @patch("src.planner.runner.find_repo_root")
    def test_main_runs_one_floorplan_for_first_n_k_values(
        self,
        mock_find_repo_root,
        mock_list_floorplans,
        mock_run_planner_batch,
    ) -> None:
        mock_find_repo_root.return_value = Path.cwd()
        mock_list_floorplans.return_value = ("ground-back", "ground-front")

        exit_code = runner.main(
            ["--floorplan", "ground-back", "--first-k-values", "2"]
        )

        self.assertEqual(exit_code, 0)
        mock_run_planner_batch.assert_called_once()
        config = mock_run_planner_batch.call_args.args[0]
        self.assertEqual(config.floorplan_name, "ground-back")
        self.assertEqual(mock_run_planner_batch.call_args.kwargs["k_values"], (10, 11))
        self.assertFalse(mock_run_planner_batch.call_args.kwargs["force"])
        self.assertIsNotNone(mock_run_planner_batch.call_args.kwargs["progress_writer"])

    @patch("src.planner.runner.run_planner_batches_for_floorplans")
    @patch("src.planner.runner.list_traced_floorplan_names")
    @patch("src.planner.runner.find_repo_root")
    def test_main_runs_all_floorplans_only_with_explicit_flag(
        self,
        mock_find_repo_root,
        mock_list_floorplans,
        mock_run_batches,
    ) -> None:
        mock_find_repo_root.return_value = Path.cwd()
        mock_list_floorplans.return_value = (
            "ground-back",
            "ground-front",
            "second-back",
            "second-front",
        )

        exit_code = runner.main(["--all-floorplans", "--all-k-values", "--force"])

        self.assertEqual(exit_code, 0)
        mock_run_batches.assert_called_once_with(
            (
                "ground-back",
                "ground-front",
                "second-back",
                "second-front",
            ),
            base_config=ANY,
            k_values=(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            force=True,
            status_log_path=None,
            progress_writer=ANY,
            max_workers=None,
        )

    @patch("src.planner.k_runner.run_planner_batch")
    def test_floorplan_batch_runner_preserves_request_order_under_parallel_completion(
        self,
        mock_run_planner_batch,
    ) -> None:
        worker_daemon_flags: list[bool] = []

        def fake_run_planner_batch(
            config,
            *,
            k_values,
            force,
            solver,
            status_log_path,
            progress_writer,
        ):
            if config.floorplan_name == "ground-back":
                time.sleep(0.05)
            worker_daemon_flags.append(threading.current_thread().daemon)
            progress_writer.write("[phase04] synthetic status line\n")
            progress_writer.flush()
            return SimpleNamespace(
                workspace=SimpleNamespace(
                    floorplan=SimpleNamespace(name=config.floorplan_name)
                )
            )

        mock_run_planner_batch.side_effect = fake_run_planner_batch
        progress_buffer = StringIO()

        results = run_planner_batches_for_floorplans(
            ("ground-back", "ground-front"),
            base_config=PlannerConfig(),
            k_values=(10,),
            progress_writer=progress_buffer,
            max_workers=2,
        )

        self.assertEqual(
            [result.workspace.floorplan.name for result in results],
            ["ground-back", "ground-front"],
        )
        self.assertTrue(all(worker_daemon_flags))

    def test_prefixed_writer_tags_each_progress_line_with_floorplan_name(self) -> None:
        base_writer = StringIO()
        prefixed_writer = _PrefixedWriter(
            base_writer,
            prefix="[floorplan=ground-back] ",
        )

        prefixed_writer.write("[phase04] synthetic status line\n")
        prefixed_writer.flush()

        self.assertEqual(
            base_writer.getvalue(),
            "[floorplan=ground-back] [phase04] synthetic status line\n",
        )

    @patch("src.planner.runner.run_planner_batches_for_floorplans")
    @patch("src.planner.runner.list_traced_floorplan_names")
    @patch("src.planner.runner.find_repo_root")
    def test_main_passes_explicit_worker_count_for_parallel_floorplan_runs(
        self,
        mock_find_repo_root,
        mock_list_floorplans,
        mock_run_batches,
    ) -> None:
        mock_find_repo_root.return_value = Path.cwd()
        mock_list_floorplans.return_value = ("ground-back", "ground-front")

        exit_code = runner.main(
            ["--all-floorplans", "--first-k-values", "2", "--workers", "4"]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run_batches.call_args.kwargs["max_workers"], 4)


if __name__ == "__main__":
    unittest.main()
