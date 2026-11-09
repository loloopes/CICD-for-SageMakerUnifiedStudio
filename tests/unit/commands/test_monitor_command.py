"""Unit tests for monitor command exit codes."""

import os
import tempfile
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from smus_cicd.cli import app
from smus_cicd.commands import monitor as monitor_mod


class TestMonitorCommandExitCodes:
    """Test exit codes for monitor command failure scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_invalid_manifest_returns_exit_code_1(self):
        """Test that monitor command returns exit code 1 when manifest file is invalid."""
        result = self.runner.invoke(app, ["monitor", "--manifest", "nonexistent.yaml"])

        # Verify exit code 1
        assert result.exit_code == 1

    def test_invalid_target_returns_exit_code_1(self):
        """Test that monitor command returns exit code 1 when target doesn't exist."""
        manifest_content = """
applicationName: test-pipeline

content:
  workflows:
    - workflowName: test-workflow
      connectionName: default.workflow_serverless

stages:
  dev:
    domain:
      name: test-domain
      region: ${DEV_DOMAIN_REGION:us-east-1}
    stage: DEV
    project:
      name: dev-project
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(manifest_content)
            temp_file = f.name

        try:
            result = self.runner.invoke(
                app, ["monitor", "--manifest", temp_file, "--targets", "nonexistent"]
            )

            # Verify exit code 1
            assert result.exit_code == 1
            assert "Target 'nonexistent' not found" in result.stderr
        finally:
            os.unlink(temp_file)

    def test_live_monitor_exits_1_when_run_transitions_to_failed(self):
        """Live monitor must fail the process if a watched run ends in FAILED."""
        call_count = {"n": 0}

        def fake_check(targets, manifest_file, output, previous_states, failed_runs=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                if failed_runs is not None:
                    failed_runs.append(
                        ("demo_training_pipeline", "run-abc", "FAILED")
                    )
                return False  # no longer running
            return False

        with (
            patch.object(monitor_mod, "_monitor_once"),
            patch.object(monitor_mod, "_check_status_changes", side_effect=fake_check),
            patch.object(monitor_mod.time, "sleep"),
            pytest.raises(typer.Exit) as exc_info,
        ):
            monitor_mod._monitor_live(None, "manifest.yaml", "TEXT")

        assert exc_info.value.exit_code == 1
