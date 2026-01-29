"""
OpenCode Adapter

Installs the Bash Gym trace capture plugin for OpenCode.
Uses OpenCode's plugin system (tool.execute.after hook).
"""

import os
import platform
from pathlib import Path
from typing import Tuple


def _get_plugins_dir() -> Path:
    """Get the OpenCode plugins directory."""
    if platform.system() == 'Windows':
        home = Path(os.environ.get("USERPROFILE", ""))
    else:
        home = Path.home()
    return home / ".config" / "opencode" / "plugins"


def _get_bashgym_dir() -> str:
    """Get the Bash Gym directory path for the plugin."""
    if platform.system() == 'Windows':
        return '${process.env.USERPROFILE}/.bashgym'
    else:
        return '${process.env.HOME}/.bashgym'


# The OpenCode plugin source code
OPENCODE_PLUGIN_SOURCE = '''/**
 * Bash Gym Trace Capture Plugin for OpenCode
 *
 * Automatically captures tool executions for training data collection.
 * Part of the Bash Gym multi-tool trace capture system.
 */

import { writeFileSync, readFileSync, existsSync, mkdirSync } from "fs";
import { execSync } from "child_process";
import { join } from "path";
import { homedir } from "os";

// Configuration
const BASHGYM_DIR = join(homedir(), ".bashgym");
const TRACES_DIR = join(BASHGYM_DIR, "traces");
const SESSION_FILE = join(BASHGYM_DIR, "current_session_id");
// Capture ALL tools for comprehensive training data
const RELEVANT_TOOLS = new Set([
  // Core file operations
  "Bash", "Edit", "Write", "Read",
  "bash", "edit", "write", "read",
  // Search tools
  "Glob", "Grep", "glob", "grep",
  // Web tools
  "WebFetch", "WebSearch", "webfetch", "websearch",
  // Task management
  "Task", "TodoWrite", "task", "todowrite",
  // User interaction
  "AskUserQuestion", "askuserquestion",
  // Notebook operations
  "NotebookEdit", "notebookedit",
  // Planning
  "EnterPlanMode", "ExitPlanMode", "enterplanmode", "exitplanmode",
  // Skills
  "Skill", "skill",
]);

// Ensure directories exist
function ensureDirs() {
  if (!existsSync(BASHGYM_DIR)) mkdirSync(BASHGYM_DIR, { recursive: true });
  if (!existsSync(TRACES_DIR)) mkdirSync(TRACES_DIR, { recursive: true });
}

// Get or create session ID
function getSessionId(): string {
  ensureDirs();

  if (existsSync(SESSION_FILE)) {
    return readFileSync(SESSION_FILE, "utf-8").trim();
  }

  const now = new Date();
  const timestamp = now.toISOString().replace(/[-:T]/g, "").slice(0, 15);
  const random = Math.random().toString(36).substring(2, 10);
  const sessionId = `${timestamp}_${random}`;

  writeFileSync(SESSION_FILE, sessionId);
  return sessionId;
}

// Get trace file path
function getTraceFile(): string {
  const sessionId = getSessionId();
  return join(TRACES_DIR, `session_${sessionId}.json`);
}

// Get repo info
function getRepoInfo(): Record<string, any> {
  const cwd = process.cwd();
  const info: Record<string, any> = {
    path: cwd,
    name: cwd.split(/[\\/]/).pop() || "unknown",
    git_remote: null,
    git_branch: null,
    is_git_repo: false,
  };

  try {
    execSync("git rev-parse --is-inside-work-tree", { cwd, stdio: "pipe" });
    info.is_git_repo = true;

    try {
      info.git_remote = execSync("git remote get-url origin", { cwd, stdio: "pipe" })
        .toString()
        .trim();
    } catch {}

    try {
      info.git_branch = execSync("git branch --show-current", { cwd, stdio: "pipe" })
        .toString()
        .trim();
    } catch {}
  } catch {}

  return info;
}

// Load existing trace
function loadTrace(): any[] {
  const traceFile = getTraceFile();
  if (!existsSync(traceFile)) return [];

  try {
    const content = readFileSync(traceFile, "utf-8");
    return content.trim() ? JSON.parse(content) : [];
  } catch {
    return [];
  }
}

// Save trace
function saveTrace(trace: any[]) {
  const traceFile = getTraceFile();
  writeFileSync(traceFile, JSON.stringify(trace, null, 2));
}

// Generate step ID
function generateStepId(): string {
  const timestamp = new Date().toISOString();
  const random = Math.random().toString(36).substring(2, 10);
  return `${timestamp}_${random}`;
}

// Main plugin export
export const BashGymTracePlugin = async (ctx: any) => {
  console.log("[BashGym] OpenCode trace capture plugin loaded");

  return {
    "tool.execute.after": async (input: any, output: any) => {
      const toolName = input.tool || input.name || "unknown";

      // Only capture relevant tools
      if (!RELEVANT_TOOLS.has(toolName)) {
        return;
      }

      // Extract command/input
      let command = "";
      if (typeof input.args === "string") {
        command = input.args;
      } else if (input.args?.command) {
        command = input.args.command;
      } else if (input.args?.content) {
        command = input.args.content;
      } else {
        command = JSON.stringify(input.args || {});
      }

      // Extract output
      let outputText = "";
      let exitCode: number | null = null;

      if (typeof output === "string") {
        outputText = output;
      } else if (output?.stdout || output?.stderr) {
        outputText = (output.stdout || "") + (output.stderr || "");
        exitCode = output.exitCode ?? output.exit_code ?? null;
      } else if (output?.content) {
        outputText = output.content;
      }

      // Create trace step
      const step = {
        step_id: generateStepId(),
        timestamp: new Date().toISOString(),
        tool_name: toolName,
        command: command.slice(0, 10000),
        output: outputText.slice(0, 10000),
        exit_code: exitCode,
        success: exitCode !== null ? exitCode === 0 : null,
        cwd: process.cwd(),
        repo: getRepoInfo(),
        source_tool: "opencode",
        metadata: {},
      };

      // Append to trace
      const trace = loadTrace();
      trace.push(step);
      saveTrace(trace);

      const repoName = step.repo.name || "unknown";
      console.log(
        `[BashGym] Captured: ${toolName} - ${command.slice(0, 50)}... (${repoName})`
      );
    },

    "session.deleted": async () => {
      // Session ended - could trigger promotion here
      console.log("[BashGym] OpenCode session ended");
    },
  };
};

// Export as default for OpenCode to pick up
export default BashGymTracePlugin;
'''


def install_opencode_plugin() -> Tuple[bool, str]:
    """
    Install Bash Gym trace capture plugin for OpenCode.

    Returns:
        Tuple of (success, message)
    """
    plugins_dir = _get_plugins_dir()

    # Create plugins directory if needed
    plugins_dir.mkdir(parents=True, exist_ok=True)

    plugin_path = plugins_dir / "bashgym-trace.ts"

    try:
        plugin_path.write_text(OPENCODE_PLUGIN_SOURCE)
        return True, f"Installed OpenCode plugin at {plugin_path}"
    except (IOError, OSError) as e:
        return False, f"Failed to install OpenCode plugin: {e}"


def uninstall_opencode_plugin() -> Tuple[bool, str]:
    """
    Uninstall Bash Gym plugin from OpenCode.

    Returns:
        Tuple of (success, message)
    """
    plugins_dir = _get_plugins_dir()
    plugin_path = plugins_dir / "bashgym-trace.ts"

    if not plugin_path.exists():
        return True, "No OpenCode plugin to remove"

    try:
        plugin_path.unlink()
        return True, f"Removed OpenCode plugin from {plugin_path}"
    except (IOError, OSError) as e:
        return False, f"Failed to remove OpenCode plugin: {e}"


def get_install_command() -> str:
    """Get manual install instructions for OpenCode plugin."""
    plugins_dir = _get_plugins_dir()
    return f"Copy bashgym-trace.ts to {plugins_dir}"
