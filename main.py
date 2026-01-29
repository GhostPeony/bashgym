#!/usr/bin/env python3
"""
Bash Gym - Main Orchestration Entry Point

A Self-Improving Agentic Development Gym that creates a flywheel:
ACT (Arena) -> VERIFY (Judge) -> SYNTHESIZE (Factory) -> TRAIN (Gym) -> DEPLOY

Usage:
    python main.py --task "Refactor utils.py to use Python 3.10 type hints"
    python main.py --batch tasks.jsonl --output results/
    python main.py --train --dataset data/training_batches/sft_batch.jsonl
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Import BashGym modules
from sandbox import SandboxManager, SandboxConfig
from agent_runner import AgentRunner, AgentConfig, TaskResult
from verifier import Verifier, VerificationConfig, VerificationResult
from trace_processor import TraceProcessor, ProcessedTrace
from data_factory import DataFactory, DataFactoryConfig
from trainer import Trainer, TrainerConfig, TrainingStrategy
from gym_env import BashGymGymEnv, GymEnvConfig
from model_router import ModelRouter, RouterConfig, RoutingStrategy


@dataclass
class BashGymConfig:
    """Main configuration for the BashGym system."""
    
    # Component configs
    sandbox_config: Optional[SandboxConfig] = None
    agent_config: Optional[AgentConfig] = None
    verifier_config: Optional[VerificationConfig] = None
    factory_config: Optional[DataFactoryConfig] = None
    trainer_config: Optional[TrainerConfig] = None
    gym_config: Optional[GymEnvConfig] = None
    router_config: Optional[RouterConfig] = None
    
    # Pipeline settings
    auto_train: bool = False
    min_gold_traces_for_training: int = 10
    
    # Output settings
    output_dir: str = "data"
    log_level: str = "INFO"


class BashGym:
    """
    Main orchestrator for the self-improving agent system.

    Coordinates the flywheel:
    1. ACT: Agent executes tasks in sandboxed environment
    2. VERIFY: Judge validates the solution
    3. SYNTHESIZE: Factory creates training data from gold traces
    4. TRAIN: Gym fine-tunes student models
    5. DEPLOY: Router progressively shifts traffic to student
    """

    def __init__(self, config: Optional[BashGymConfig] = None):
        """Initialize the BashGym system."""
        self.config = config or BashGymConfig()
        
        # Initialize components
        self.sandbox_manager = SandboxManager(self.config.sandbox_config)
        self.agent_runner = AgentRunner(
            self.config.agent_config,
            self.sandbox_manager
        )
        self.verifier = Verifier(self.config.verifier_config)
        self.trace_processor = TraceProcessor()
        self.data_factory = DataFactory(self.config.factory_config)
        self.trainer = Trainer(self.config.trainer_config)
        self.gym_env = BashGymGymEnv(
            self.config.gym_config,
            self.sandbox_manager,
            self.verifier
        )
        self.model_router = ModelRouter(self.config.router_config)
        
        # Ensure directories exist
        self._setup_directories()
        
        # Statistics
        self.stats = {
            "tasks_executed": 0,
            "tasks_verified": 0,
            "gold_traces": 0,
            "failed_traces": 0,
            "training_runs": 0
        }
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        dirs = [
            "data/gold_traces",
            "data/failed_traces",
            "data/training_batches",
            "data/models",
            "data/verification_results",
            "data/trajectories",
            "data/routing_metrics"
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
    
    async def run_task(
        self,
        task_prompt: str,
        task_id: Optional[str] = None,
        repository_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single task through the full pipeline.
        
        Args:
            task_prompt: The task description
            task_id: Optional unique identifier
            repository_url: Optional git repo to work on
            
        Returns:
            Dict with task results and pipeline status
        """
        task_id = task_id or f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"OUROBOROS: Starting task {task_id}")
        print(f"{'='*60}")
        print(f"Task: {task_prompt[:100]}...")
        
        result = {
            "task_id": task_id,
            "task_prompt": task_prompt,
            "stages": {}
        }
        
        # Stage 1: ACT - Execute the task
        print("\n[1/4] ACT: Executing task in sandbox...")
        task_result = self.agent_runner.run_task(
            task_prompt=task_prompt,
            task_id=task_id,
            repository_url=repository_url,
            on_output=lambda x: print(f"  > {x.strip()}")
        )
        
        result["stages"]["act"] = {
            "success": task_result.success,
            "exit_code": task_result.exit_code,
            "duration_seconds": task_result.duration_seconds,
            "trace_path": str(task_result.trace_path) if task_result.trace_path else None
        }
        self.stats["tasks_executed"] += 1
        
        if not task_result.success:
            print(f"  ✗ Task execution failed (exit code: {task_result.exit_code})")
            result["success"] = False
            return result
        
        print(f"  ✓ Task completed in {task_result.duration_seconds:.1f}s")
        
        # Stage 2: VERIFY - Validate the solution
        print("\n[2/4] VERIFY: Running verification tests...")
        workspace_path = Path(task_result.metadata.get("workspace_path", "."))
        
        verification_result = self.verifier.verify(
            workspace_path=workspace_path,
            task_id=task_id
        )
        
        result["stages"]["verify"] = {
            "status": verification_result.status.value,
            "passed_tests": verification_result.passed_tests,
            "failed_tests": verification_result.failed_tests,
            "total_tests": verification_result.total_tests
        }
        self.stats["tasks_verified"] += 1
        
        if verification_result.success:
            print(f"  ✓ Verification passed ({verification_result.passed_tests}/{verification_result.total_tests} tests)")
            trace_dest = Path("data/gold_traces") / f"{task_id}.json"
            self.stats["gold_traces"] += 1
        else:
            print(f"  ✗ Verification failed ({verification_result.failed_tests} failures)")
            trace_dest = Path("data/failed_traces") / f"{task_id}.json"
            self.stats["failed_traces"] += 1
        
        # Copy trace to appropriate directory
        if task_result.trace_path and task_result.trace_path.exists():
            import shutil
            shutil.copy(task_result.trace_path, trace_dest)
            result["trace_path"] = str(trace_dest)
        
        # Stage 3: SYNTHESIZE - Create training data (if verified)
        if verification_result.success:
            print("\n[3/4] SYNTHESIZE: Processing trace for training...")
            
            processed = self.trace_processor.process_trace(trace_dest)
            if processed:
                example = self.data_factory.process_gold_trace(trace_dest)
                if example:
                    result["stages"]["synthesize"] = {
                        "example_id": example.example_id,
                        "quality_score": processed.quality_metrics.quality_score
                    }
                    print(f"  ✓ Created training example (quality: {processed.quality_metrics.quality_score:.2f})")
                else:
                    print("  ⚠ Could not create training example")
            else:
                print("  ⚠ Trace did not meet quality threshold")
        else:
            print("\n[3/4] SYNTHESIZE: Skipped (verification failed)")
            result["stages"]["synthesize"] = {"skipped": True}
        
        # Stage 4: TRAIN - Trigger training if enough data
        if self.config.auto_train and self.stats["gold_traces"] >= self.config.min_gold_traces_for_training:
            print("\n[4/4] TRAIN: Triggering training run...")
            await self._trigger_training()
            result["stages"]["train"] = {"triggered": True}
        else:
            print(f"\n[4/4] TRAIN: Waiting for more data ({self.stats['gold_traces']}/{self.config.min_gold_traces_for_training} traces)")
            result["stages"]["train"] = {"waiting": True}
        
        result["success"] = verification_result.success
        
        print(f"\n{'='*60}")
        print(f"OUROBOROS: Task {task_id} completed")
        print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"{'='*60}\n")
        
        return result
    
    async def run_batch(
        self,
        tasks_file: Path,
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a batch of tasks from a JSONL file.
        
        Args:
            tasks_file: Path to JSONL file with tasks
            output_dir: Directory to save results
            
        Returns:
            List of task results
        """
        output_dir = output_dir or Path("data/batch_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        with open(tasks_file, 'r') as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        
        print(f"Running batch of {len(tasks)} tasks...")
        
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] Processing task...")
            
            result = await self.run_task(
                task_prompt=task.get("prompt", task.get("task", "")),
                task_id=task.get("id"),
                repository_url=task.get("repository_url")
            )
            results.append(result)
        
        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"batch_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        success_count = sum(1 for r in results if r.get("success"))
        print(f"\nBatch complete: {success_count}/{len(results)} tasks succeeded")
        print(f"Results saved to: {results_file}")
        
        return results
    
    async def _trigger_training(self) -> None:
        """Trigger a training run with accumulated data."""
        # Process all gold traces
        gold_dir = Path("data/gold_traces")
        failed_dir = Path("data/failed_traces")
        
        examples, dpo_examples = await self.data_factory.process_trace_directory(
            gold_dir, failed_dir
        )
        
        if examples:
            # Save training batch
            batch_path = self.data_factory.save_training_batch(examples, "sft")
            
            # Run SFT training
            run = self.trainer.train_sft(
                dataset_path=batch_path,
                callback=lambda m: print(f"  Training: Epoch {m['epoch']}/{m['total_epochs']}, Loss: {m['loss']:.4f}")
            )
            
            self.stats["training_runs"] += 1
            print(f"  ✓ Training run {run.run_id} completed")
            
            # Update router with new model performance
            if run.status == "completed":
                self.model_router.update_student_performance(0.8)  # Placeholder
        
        if dpo_examples:
            # Save DPO batch
            dpo_path = self.data_factory.save_dpo_batch(dpo_examples, "dpo")
            print(f"  ✓ Saved {len(dpo_examples)} DPO examples")
    
    async def train(
        self,
        dataset_path: Path,
        strategy: TrainingStrategy = TrainingStrategy.SFT
    ) -> Dict[str, Any]:
        """
        Run training on a specific dataset.
        
        Args:
            dataset_path: Path to training data
            strategy: Training strategy to use
            
        Returns:
            Training run results
        """
        print(f"\nStarting {strategy.value} training on {dataset_path}...")
        
        if strategy == TrainingStrategy.SFT:
            run = self.trainer.train_sft(
                dataset_path=dataset_path,
                callback=lambda m: print(f"  Epoch {m['epoch']}: Loss={m['loss']:.4f}")
            )
        elif strategy == TrainingStrategy.DPO:
            run = self.trainer.train_dpo(
                dataset_path=dataset_path,
                callback=lambda m: print(f"  Epoch {m['epoch']}: Loss={m['loss']:.4f}")
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        return run.to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            **self.stats,
            "routing": self.model_router.get_routing_stats() if self.model_router.routing_history else {}
        }
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        self.sandbox_manager.cleanup_all()
        await self.data_factory.close()
        await self.model_router.close()
        self.gym_env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bash Gym - Self-Improving Agentic Development Gym"
    )
    
    # Task execution
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Single task to execute"
    )
    parser.add_argument(
        "--batch", "-b",
        type=Path,
        help="JSONL file with batch of tasks"
    )
    parser.add_argument(
        "--repo", "-r",
        type=str,
        help="Git repository URL to work on"
    )
    
    # Training
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training mode"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        help="Training dataset path"
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        choices=["sft", "dpo", "grpo"],
        default="sft",
        help="Training strategy"
    )
    
    # Configuration
    parser.add_argument(
        "--auto-train",
        action="store_true",
        help="Automatically trigger training when enough data"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data"),
        help="Output directory"
    )
    
    # Info
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show system statistics"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    config = BashGymConfig(
        auto_train=args.auto_train,
        output_dir=str(args.output)
    )

    # Initialize system
    bashgym = BashGym(config)
    
    try:
        if args.task:
            # Single task execution
            result = await bashgym.run_task(
                task_prompt=args.task,
                repository_url=args.repo
            )
            print(json.dumps(result, indent=2))

        elif args.batch:
            # Batch execution
            results = await bashgym.run_batch(
                tasks_file=args.batch,
                output_dir=args.output / "batch_results"
            )

        elif args.train:
            # Training mode
            if not args.dataset:
                print("Error: --dataset required for training mode")
                sys.exit(1)

            strategy = TrainingStrategy(args.strategy)
            result = await bashgym.train(args.dataset, strategy)
            print(json.dumps(result, indent=2))

        elif args.stats:
            # Show statistics
            stats = bashgym.get_stats()
            print(json.dumps(stats, indent=2))

        else:
            # Interactive mode or show help
            print("Bash Gym - Self-Improving Agentic Development Gym")
            print("\nUsage examples:")
            print('  python main.py --task "Fix the bug in utils.py"')
            print('  python main.py --batch tasks.jsonl')
            print('  python main.py --train --dataset data/training_batches/sft.jsonl')
            print("\nRun with --help for all options.")

    finally:
        await bashgym.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
