"""
Verifier for The Judge

Executes verification tests to determine if an agent's solution is correct.
Supports pytest, bats, and custom verification scripts.

Extended with NeMo Evaluator integration for comprehensive model evaluation.
"""

import os
import json
import subprocess
import shutil
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from datetime import datetime, timezone
from enum import Enum

if TYPE_CHECKING:
    from .evaluator import EvaluatorClient


class VerificationStatus(Enum):
    """Status of a verification run."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class VerificationConfig:
    """Configuration for the verifier."""
    
    # Test execution settings
    timeout: int = 300  # 5 minutes
    max_retries: int = 1
    
    # Test discovery
    test_patterns: List[str] = field(default_factory=lambda: [
        "test_*.py",
        "*_test.py",
        "tests/*.py",
        "verify.sh",
        "verify.py"
    ])
    
    # Pytest settings
    pytest_args: List[str] = field(default_factory=lambda: [
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])
    
    # Output settings
    capture_output: bool = True
    save_results: bool = True
    results_dir: str = "data/verification_results"


@dataclass
class VerificationResult:
    """Result of a verification run."""
    
    task_id: str
    status: VerificationStatus
    exit_code: int
    passed_tests: int
    failed_tests: int
    total_tests: int
    duration_seconds: float
    stdout: str
    stderr: str
    test_details: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "total_tests": self.total_tests,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "test_details": self.test_details,
            "error_message": self.error_message
        }


class Verifier:
    """
    Executes verification tests for agent solutions.

    The Judge determines if a task was successfully completed by:
    1. Running pytest/bats tests
    2. Executing custom verify.sh scripts
    3. Checking exit codes and test results
    4. (Optional) Running NeMo Evaluator for comprehensive metrics

    Only exit code 0 triggers the Data Designer pipeline.
    Failed trajectories are stored for DPO training.
    """

    def __init__(
        self,
        config: Optional[VerificationConfig] = None,
        evaluator_client: Optional["EvaluatorClient"] = None
    ):
        """Initialize the verifier.

        Args:
            config: Verification configuration
            evaluator_client: Optional NeMo Evaluator client for extended evaluation
        """
        self.config = config or VerificationConfig()
        self.evaluator_client = evaluator_client
    
    def verify(
        self,
        workspace_path: Path,
        task_id: str,
        sandbox_manager=None,
        sandbox_id: Optional[str] = None
    ) -> VerificationResult:
        """
        Run verification tests for a task.
        
        Args:
            workspace_path: Path to the workspace
            task_id: Unique task identifier
            sandbox_manager: Optional sandbox manager for isolated execution
            sandbox_id: Optional sandbox ID for isolated execution
            
        Returns:
            VerificationResult with test outcomes
        """
        start_time = datetime.now(timezone.utc)
        workspace = Path(workspace_path)
        
        # Discover verification method
        verify_script = self._find_verify_script(workspace)
        test_files = self._find_test_files(workspace)
        
        if verify_script:
            # Run custom verification script
            result = self._run_verify_script(
                verify_script, workspace, task_id, sandbox_manager, sandbox_id
            )
        elif test_files:
            # Run pytest
            result = self._run_pytest(
                test_files, workspace, task_id, sandbox_manager, sandbox_id
            )
        else:
            # No tests found - skip verification
            result = VerificationResult(
                task_id=task_id,
                status=VerificationStatus.SKIPPED,
                exit_code=0,
                passed_tests=0,
                failed_tests=0,
                total_tests=0,
                duration_seconds=0,
                stdout="No verification tests found",
                stderr=""
            )
        
        # Calculate duration
        end_time = datetime.now(timezone.utc)
        result.duration_seconds = (end_time - start_time).total_seconds()
        
        # Create verification flag file
        self._create_verification_flag(workspace, result)
        
        # Save results
        if self.config.save_results:
            self._save_results(workspace, result)
        
        return result
    
    def _find_verify_script(self, workspace: Path) -> Optional[Path]:
        """Find a custom verification script."""
        for script_name in ["verify.sh", "verify.py", "verification.sh"]:
            script_path = workspace / script_name
            if script_path.exists():
                return script_path
        return None
    
    def _find_test_files(self, workspace: Path) -> List[Path]:
        """Find test files in the workspace."""
        test_files = []
        
        for pattern in self.config.test_patterns:
            if pattern.endswith(".sh"):
                continue  # Skip shell scripts, handled separately
            test_files.extend(workspace.glob(pattern))
            test_files.extend(workspace.glob(f"**/{pattern}"))
        
        # Deduplicate and filter
        seen = set()
        unique_files = []
        for f in test_files:
            if f not in seen and f.is_file():
                seen.add(f)
                unique_files.append(f)
        
        return unique_files
    
    def _run_verify_script(
        self,
        script_path: Path,
        workspace: Path,
        task_id: str,
        sandbox_manager=None,
        sandbox_id: Optional[str] = None
    ) -> VerificationResult:
        """Run a custom verification script."""
        try:
            # Determine how to run the script
            if script_path.suffix == ".sh":
                cmd = ["bash", str(script_path)]
            elif script_path.suffix == ".py":
                cmd = ["python", str(script_path)]
            else:
                cmd = [str(script_path)]
            
            # Execute in sandbox if available
            if sandbox_manager and sandbox_id:
                result = sandbox_manager.execute_command(
                    sandbox_id,
                    " ".join(cmd),
                    timeout=self.config.timeout
                )
                exit_code = result["exit_code"]
                stdout = result["stdout"]
                stderr = result["stderr"]
            else:
                # Execute locally
                process = subprocess.run(
                    cmd,
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                exit_code = process.returncode
                stdout = process.stdout
                stderr = process.stderr
            
            # Determine status
            if exit_code == 0:
                status = VerificationStatus.PASSED
                passed = 1
                failed = 0
            else:
                status = VerificationStatus.FAILED
                passed = 0
                failed = 1
            
            return VerificationResult(
                task_id=task_id,
                status=status,
                exit_code=exit_code,
                passed_tests=passed,
                failed_tests=failed,
                total_tests=1,
                duration_seconds=0,
                stdout=stdout,
                stderr=stderr
            )
            
        except subprocess.TimeoutExpired:
            return VerificationResult(
                task_id=task_id,
                status=VerificationStatus.TIMEOUT,
                exit_code=-1,
                passed_tests=0,
                failed_tests=0,
                total_tests=1,
                duration_seconds=self.config.timeout,
                stdout="",
                stderr="Verification timed out",
                error_message="Timeout"
            )
        except Exception as e:
            return VerificationResult(
                task_id=task_id,
                status=VerificationStatus.ERROR,
                exit_code=-1,
                passed_tests=0,
                failed_tests=0,
                total_tests=0,
                duration_seconds=0,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
    
    def _run_pytest(
        self,
        test_files: List[Path],
        workspace: Path,
        task_id: str,
        sandbox_manager=None,
        sandbox_id: Optional[str] = None
    ) -> VerificationResult:
        """Run pytest on test files."""
        try:
            # Build pytest command
            cmd = ["python", "-m", "pytest"]
            cmd.extend(self.config.pytest_args)
            cmd.extend(["--json-report", "--json-report-file=.pytest_report.json"])
            cmd.extend([str(f.relative_to(workspace)) for f in test_files[:10]])  # Limit files
            
            # Execute
            if sandbox_manager and sandbox_id:
                result = sandbox_manager.execute_command(
                    sandbox_id,
                    " ".join(cmd),
                    timeout=self.config.timeout
                )
                exit_code = result["exit_code"]
                stdout = result["stdout"]
                stderr = result["stderr"]
            else:
                process = subprocess.run(
                    cmd,
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                exit_code = process.returncode
                stdout = process.stdout
                stderr = process.stderr
            
            # Parse pytest output
            passed, failed, total, details = self._parse_pytest_output(
                workspace, stdout
            )
            
            # Determine status
            if exit_code == 0:
                status = VerificationStatus.PASSED
            else:
                status = VerificationStatus.FAILED
            
            return VerificationResult(
                task_id=task_id,
                status=status,
                exit_code=exit_code,
                passed_tests=passed,
                failed_tests=failed,
                total_tests=total,
                duration_seconds=0,
                stdout=stdout,
                stderr=stderr,
                test_details=details
            )
            
        except subprocess.TimeoutExpired:
            return VerificationResult(
                task_id=task_id,
                status=VerificationStatus.TIMEOUT,
                exit_code=-1,
                passed_tests=0,
                failed_tests=0,
                total_tests=len(test_files),
                duration_seconds=self.config.timeout,
                stdout="",
                stderr="Pytest timed out",
                error_message="Timeout"
            )
        except Exception as e:
            return VerificationResult(
                task_id=task_id,
                status=VerificationStatus.ERROR,
                exit_code=-1,
                passed_tests=0,
                failed_tests=0,
                total_tests=0,
                duration_seconds=0,
                stdout="",
                stderr=str(e),
                error_message=str(e)
            )
    
    def _parse_pytest_output(
        self,
        workspace: Path,
        stdout: str
    ) -> Tuple[int, int, int, List[Dict[str, Any]]]:
        """Parse pytest output for test results."""
        passed = 0
        failed = 0
        total = 0
        details = []
        
        # Try to read JSON report
        report_path = workspace / ".pytest_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                
                summary = report.get("summary", {})
                passed = summary.get("passed", 0)
                failed = summary.get("failed", 0)
                total = summary.get("total", passed + failed)
                
                for test in report.get("tests", []):
                    details.append({
                        "name": test.get("nodeid"),
                        "outcome": test.get("outcome"),
                        "duration": test.get("duration")
                    })
                
                return passed, failed, total, details
            except:
                pass
        
        # Fallback: parse stdout
        for line in stdout.split("\n"):
            if " passed" in line:
                try:
                    passed = int(line.split()[0])
                except:
                    pass
            if " failed" in line:
                try:
                    failed = int(line.split()[0])
                except:
                    pass
        
        total = passed + failed
        return passed, failed, total, details
    
    def _create_verification_flag(
        self,
        workspace: Path,
        result: VerificationResult
    ) -> None:
        """Create verification flag file based on result."""
        if result.success:
            flag_path = workspace / ".verification_passed"
            flag_path.write_text(json.dumps({
                "task_id": result.task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "passed_tests": result.passed_tests,
                "total_tests": result.total_tests
            }))
        else:
            # Remove any existing flag
            flag_path = workspace / ".verification_passed"
            if flag_path.exists():
                flag_path.unlink()
    
    def _save_results(self, workspace: Path, result: VerificationResult) -> None:
        """Save verification results to file."""
        results_dir = workspace / self.config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_dir / f"verification_{result.task_id}.json"
        result_file.write_text(json.dumps(result.to_dict(), indent=2))

    async def evaluate_with_llm_judge(
        self,
        task_prompt: str,
        agent_response: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an agent response using LLM-as-Judge.

        Requires evaluator_client to be configured.

        Args:
            task_prompt: The original task/prompt
            agent_response: The agent's response to evaluate
            metrics: List of metrics to score (default: all)

        Returns:
            Dictionary of metric scores and reasoning
        """
        if not self.evaluator_client:
            return {"error": "Evaluator client not configured"}

        try:
            scores = await self.evaluator_client.judge_response(
                prompt=task_prompt,
                response=agent_response,
                metrics=metrics
            )

            return {
                "scores": {s.metric: s.score for s in scores},
                "details": [s.to_dict() for s in scores],
                "average_score": sum(s.score for s in scores) / max(len(scores), 1)
            }

        except Exception as e:
            return {"error": str(e)}

    async def evaluate_agentic_trace(
        self,
        trace: List[Dict[str, Any]],
        task_description: str
    ) -> Dict[str, float]:
        """
        Evaluate an agentic execution trace for quality metrics.

        Requires evaluator_client to be configured.

        Args:
            trace: List of action-observation pairs
            task_description: The original task

        Returns:
            Dictionary of agentic metrics
        """
        if not self.evaluator_client:
            return {"error": -1.0}

        try:
            return await self.evaluator_client.evaluate_agentic_trace(
                trace=trace,
                task_description=task_description
            )

        except Exception as e:
            return {"error": -1.0, "message": str(e)}

    async def run_benchmark_evaluation(
        self,
        model_path: str,
        benchmarks: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run benchmark evaluation on a model.

        Requires evaluator_client to be configured.

        Args:
            model_path: Path to the model or model identifier
            benchmarks: List of benchmarks to run

        Returns:
            List of evaluation results
        """
        if not self.evaluator_client:
            return [{"error": "Evaluator client not configured"}]

        try:
            results = await self.evaluator_client.evaluate_model(
                model_path=model_path,
                benchmarks=benchmarks
            )
            return [r.to_dict() for r in results]

        except Exception as e:
            return [{"error": str(e)}]


# Default verification script template
VERIFY_SCRIPT_TEMPLATE = """#!/bin/bash
# Verification script for Bash Gym
# Exit code 0 = success, non-zero = failure

set -e

echo "Running verification tests..."

# Run pytest if available
if command -v pytest &> /dev/null; then
    pytest -v --tb=short
    exit $?
fi

# Run python tests
if [ -f "test_*.py" ] || [ -d "tests" ]; then
    python -m pytest -v
    exit $?
fi

# Custom verification logic here
echo "No tests found, assuming success"
exit 0
"""
