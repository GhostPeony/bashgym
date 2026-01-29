"""
Sandbox Manager for The Arena

Manages Docker containers for isolated agent execution. Provides
network isolation, resource limits, and secure workspace management.
"""

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import docker
from docker.types import Mount
from docker.errors import ContainerError, ImageNotFound, APIError


@dataclass
class SandboxConfig:
    """Configuration for sandbox containers."""
    
    # Container settings
    image: str = "python:3.10-slim"
    memory_limit: str = "2g"
    cpu_limit: float = 2.0
    timeout: int = 3600  # 1 hour max
    
    # Network settings
    network_mode: str = "none"  # Network isolated by default
    allowed_hosts: List[str] = field(default_factory=lambda: [
        "pypi.org",
        "files.pythonhosted.org",
        "registry.npmjs.org"
    ])
    
    # Security settings
    read_only_root: bool = False
    privileged: bool = False
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=lambda: ["CHOWN", "SETUID", "SETGID"])
    
    # Workspace settings
    workspace_base: str = "/tmp/bashgym_workspaces"
    mount_hooks: bool = True
    
    # Resource paths
    hooks_path: Optional[str] = None
    repository_url: Optional[str] = None


@dataclass
class SandboxInstance:
    """Represents a running sandbox instance."""
    
    sandbox_id: str
    container_id: str
    workspace_path: Path
    container: Any  # docker.models.containers.Container
    config: SandboxConfig
    status: str = "created"


class SandboxManager:
    """
    Manages Docker sandboxes for agent execution.
    
    Provides isolated environments with:
    - Network restrictions
    - Resource limits
    - Workspace isolation
    - Claude Code hooks integration
    """
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize the sandbox manager."""
        self.config = config or SandboxConfig()
        self.client = docker.from_env()
        self.active_sandboxes: Dict[str, SandboxInstance] = {}
        
        # Ensure workspace base exists
        Path(self.config.workspace_base).mkdir(parents=True, exist_ok=True)
    
    def create_sandbox(
        self,
        task_id: Optional[str] = None,
        repository_url: Optional[str] = None,
        initial_files: Optional[Dict[str, str]] = None
    ) -> SandboxInstance:
        """
        Create a new isolated sandbox for agent execution.
        
        Args:
            task_id: Optional task identifier
            repository_url: Git repository to clone into sandbox
            initial_files: Dict of {filename: content} to create
            
        Returns:
            SandboxInstance with container and workspace info
        """
        # Generate unique sandbox ID
        sandbox_id = task_id or f"sandbox_{uuid.uuid4().hex[:12]}"
        
        # Create workspace directory
        workspace_path = Path(self.config.workspace_base) / sandbox_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Set up hooks directory in workspace
        hooks_dest = workspace_path / ".claude" / "hooks"
        hooks_dest.mkdir(parents=True, exist_ok=True)
        
        # Copy hooks if available
        if self.config.hooks_path and Path(self.config.hooks_path).exists():
            for hook_file in Path(self.config.hooks_path).glob("*.py"):
                shutil.copy(hook_file, hooks_dest / hook_file.name)
        
        # Create initial files
        if initial_files:
            for filename, content in initial_files.items():
                file_path = workspace_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
        
        # Prepare container mounts
        mounts = [
            Mount(
                target="/workspace",
                source=str(workspace_path),
                type="bind",
                read_only=False
            )
        ]
        
        # Container environment
        environment = {
            "OUROBOROS_SANDBOX_ID": sandbox_id,
            "OUROBOROS_WORKSPACE": "/workspace",
            "PYTHONUNBUFFERED": "1",
            "HOME": "/workspace"
        }
        
        # Create container
        try:
            container = self.client.containers.create(
                image=self.config.image,
                name=f"bashgym_{sandbox_id}",
                mounts=mounts,
                environment=environment,
                working_dir="/workspace",
                network_mode=self.config.network_mode,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                cap_drop=self.config.cap_drop,
                cap_add=self.config.cap_add,
                read_only=self.config.read_only_root,
                privileged=self.config.privileged,
                stdin_open=True,
                tty=True,
                detach=True,
                command="/bin/bash"
            )
        except ImageNotFound:
            # Pull image if not found
            print(f"Pulling image {self.config.image}...")
            self.client.images.pull(self.config.image)
            container = self.client.containers.create(
                image=self.config.image,
                name=f"bashgym_{sandbox_id}",
                mounts=mounts,
                environment=environment,
                working_dir="/workspace",
                network_mode=self.config.network_mode,
                mem_limit=self.config.memory_limit,
                nano_cpus=int(self.config.cpu_limit * 1e9),
                command="/bin/bash",
                stdin_open=True,
                tty=True,
                detach=True
            )
        
        # Create sandbox instance
        sandbox = SandboxInstance(
            sandbox_id=sandbox_id,
            container_id=container.id,
            workspace_path=workspace_path,
            container=container,
            config=self.config,
            status="created"
        )
        
        self.active_sandboxes[sandbox_id] = sandbox
        return sandbox
    
    def start_sandbox(self, sandbox_id: str) -> bool:
        """Start a created sandbox container."""
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox = self.active_sandboxes[sandbox_id]
        try:
            sandbox.container.start()
            sandbox.status = "running"
            return True
        except APIError as e:
            print(f"Error starting sandbox: {e}")
            return False
    
    def execute_command(
        self,
        sandbox_id: str,
        command: str,
        timeout: int = 300,
        workdir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a command inside the sandbox.
        
        Args:
            sandbox_id: ID of the sandbox
            command: Command to execute
            timeout: Command timeout in seconds
            workdir: Working directory (default: /workspace)
            
        Returns:
            Dict with exit_code, stdout, stderr
        """
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox = self.active_sandboxes[sandbox_id]
        
        if sandbox.status != "running":
            self.start_sandbox(sandbox_id)
        
        # Check for dangerous commands
        if self._is_dangerous_command(command):
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": "BLOCKED: Dangerous command detected",
                "blocked": True
            }
        
        try:
            exec_result = sandbox.container.exec_run(
                cmd=["bash", "-c", command],
                workdir=workdir or "/workspace",
                demux=True,
                tty=False
            )
            
            stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
            stderr = exec_result.output[1].decode() if exec_result.output[1] else ""
            
            return {
                "exit_code": exec_result.exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "blocked": False
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "blocked": False
            }
    
    def clone_repository(self, sandbox_id: str, repo_url: str, branch: str = "main") -> bool:
        """Clone a git repository into the sandbox workspace."""
        result = self.execute_command(
            sandbox_id,
            f"git clone --depth 1 --branch {branch} {repo_url} .",
            timeout=300
        )
        return result["exit_code"] == 0
    
    def get_file_tree(self, sandbox_id: str, path: str = "/workspace") -> str:
        """Get the file tree of the sandbox workspace."""
        result = self.execute_command(
            sandbox_id,
            f"find {path} -type f -name '*.py' -o -name '*.js' -o -name '*.ts' | head -100"
        )
        return result["stdout"]
    
    def stop_sandbox(self, sandbox_id: str, remove: bool = False) -> bool:
        """Stop a running sandbox."""
        if sandbox_id not in self.active_sandboxes:
            return False
        
        sandbox = self.active_sandboxes[sandbox_id]
        try:
            sandbox.container.stop(timeout=10)
            sandbox.status = "stopped"
            
            if remove:
                sandbox.container.remove()
                del self.active_sandboxes[sandbox_id]
            
            return True
        except Exception as e:
            print(f"Error stopping sandbox: {e}")
            return False
    
    def cleanup_sandbox(self, sandbox_id: str, remove_workspace: bool = False) -> bool:
        """Clean up a sandbox completely."""
        if sandbox_id not in self.active_sandboxes:
            return False
        
        sandbox = self.active_sandboxes[sandbox_id]
        
        try:
            # Stop and remove container
            try:
                sandbox.container.stop(timeout=5)
            except:
                pass
            
            try:
                sandbox.container.remove(force=True)
            except:
                pass
            
            # Remove workspace if requested
            if remove_workspace and sandbox.workspace_path.exists():
                shutil.rmtree(sandbox.workspace_path)
            
            del self.active_sandboxes[sandbox_id]
            return True
        except Exception as e:
            print(f"Error cleaning up sandbox: {e}")
            return False
    
    def cleanup_all(self) -> None:
        """Clean up all active sandboxes."""
        for sandbox_id in list(self.active_sandboxes.keys()):
            self.cleanup_sandbox(sandbox_id, remove_workspace=True)
    
    def _is_dangerous_command(self, command: str) -> bool:
        """Check if a command is potentially dangerous."""
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf /*",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # Fork bomb
            "chmod -R 777 /",
            "chown -R",
            "> /dev/sda",
            "curl | bash",
            "wget | bash",
            "nc -e",
            "python -c.*socket",
        ]
        
        command_lower = command.lower().replace(" ", "")
        for pattern in dangerous_patterns:
            if pattern.replace(" ", "") in command_lower:
                return True
        
        return False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all sandboxes."""
        self.cleanup_all()
