"""
Trace Capture Setup

CLI and API for setting up trace capture across multiple AI coding tools.
Auto-detects installed tools and configures appropriate adapters.
"""

import sys
from typing import List, Dict, Any, Optional

from .detector import detect_tools, ToolInfo, get_tool_status
from .adapters import (
    install_claude_code_hooks,
    uninstall_claude_code_hooks,
    install_opencode_plugin,
    uninstall_opencode_plugin,
)
from .importers import import_today, import_recent, import_session


def setup_trace_capture(
    tools: Optional[List[str]] = None,
    auto_detect: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Set up trace capture for AI coding tools.

    Args:
        tools: Specific tools to set up (e.g., ["claude_code", "opencode"]).
               If None and auto_detect is True, will detect and set up all.
        auto_detect: Automatically detect and set up installed tools.
        verbose: Print status messages.

    Returns:
        Dict with setup results for each tool.
    """
    results = {
        "success": True,
        "tools": {},
        "errors": []
    }

    # Determine which tools to set up
    if tools:
        target_tools = tools
    elif auto_detect:
        detected = detect_tools()
        target_tools = [t.adapter_type for t in detected if t.installed]
    else:
        results["success"] = False
        results["errors"].append("No tools specified and auto_detect is False")
        return results

    if verbose:
        print(f"[BashGym] Setting up trace capture for: {', '.join(target_tools)}")

    # Install adapters for each tool
    for tool in target_tools:
        if tool == "claude_code":
            success, message = install_claude_code_hooks()
            results["tools"]["claude_code"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Claude Code: {message}")
            if not success:
                results["success"] = False
                results["errors"].append(f"Claude Code: {message}")

        elif tool == "opencode":
            success, message = install_opencode_plugin()
            results["tools"]["opencode"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} OpenCode: {message}")
            if not success:
                results["success"] = False
                results["errors"].append(f"OpenCode: {message}")

        elif tool in ["aider", "continue", "cursor"]:
            results["tools"][tool] = {
                "success": False,
                "message": f"{tool} adapter not yet implemented"
            }
            if verbose:
                print(f"  ⚠ {tool}: Adapter not yet implemented")

        else:
            results["tools"][tool] = {
                "success": False,
                "message": f"Unknown tool: {tool}"
            }
            if verbose:
                print(f"  ✗ {tool}: Unknown tool")

    return results


def uninstall_trace_capture(
    tools: Optional[List[str]] = None,
    all_tools: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Uninstall trace capture from AI coding tools.

    Args:
        tools: Specific tools to uninstall from.
        all_tools: Uninstall from all known tools.
        verbose: Print status messages.

    Returns:
        Dict with uninstall results for each tool.
    """
    results = {
        "success": True,
        "tools": {},
        "errors": []
    }

    if all_tools:
        target_tools = ["claude_code", "opencode"]
    elif tools:
        target_tools = tools
    else:
        results["success"] = False
        results["errors"].append("No tools specified")
        return results

    if verbose:
        print(f"[BashGym] Uninstalling trace capture from: {', '.join(target_tools)}")

    for tool in target_tools:
        if tool == "claude_code":
            success, message = uninstall_claude_code_hooks()
            results["tools"]["claude_code"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Claude Code: {message}")

        elif tool == "opencode":
            success, message = uninstall_opencode_plugin()
            results["tools"]["opencode"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} OpenCode: {message}")

    return results


def check_setup_status(verbose: bool = True) -> Dict[str, Any]:
    """
    Check the current setup status of all tools.

    Returns:
        Dict with status of each detected tool.
    """
    status = get_tool_status()

    if verbose:
        print("[BashGym] Trace Capture Status")
        print("=" * 40)

        for tool in status["tools"]:
            installed_icon = "✓" if tool["installed"] else "✗"
            hooks_icon = "✓" if tool["hooks_installed"] else "✗"

            print(f"\n{tool['name']}:")
            print(f"  Installed: {installed_icon}")
            print(f"  Hooks configured: {hooks_icon}")
            if tool["hooks_path"]:
                print(f"  Hooks path: {tool['hooks_path']}")

        print("\n" + "=" * 40)
        summary = status["summary"]
        print(f"Total installed: {summary['installed_count']}")
        print(f"Total configured: {summary['configured_count']}")

    return status


def main():
    """CLI entry point for trace capture setup."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bash Gym Trace Capture Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bashgym-setup                    # Auto-detect and setup all tools
  bashgym-setup --tools claude     # Setup Claude Code only
  bashgym-setup --tools opencode   # Setup OpenCode only
  bashgym-setup --status           # Check current status
  bashgym-setup --uninstall        # Remove all hooks

  # Import Claude Code session history
  bashgym-setup import-today              # Import today's sessions
  bashgym-setup import-recent             # Import last 60 days
  bashgym-setup import-recent --days 30   # Import last 30 days
  bashgym-setup import-session <path>     # Import specific session file
        """
    )

    # Create subparsers for import commands
    subparsers = parser.add_subparsers(dest="command", help="Import commands")

    # import-today subcommand
    import_today_parser = subparsers.add_parser(
        "import-today",
        help="Import today's Claude Code sessions"
    )
    import_today_parser.add_argument(
        "--project", "-p",
        help="Only import sessions from projects matching this substring"
    )

    # import-recent subcommand
    import_recent_parser = subparsers.add_parser(
        "import-recent",
        help="Import recent Claude Code sessions"
    )
    import_recent_parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of days to look back (default: 60)"
    )
    import_recent_parser.add_argument(
        "--project", "-p",
        help="Only import sessions from projects matching this substring"
    )

    # import-session subcommand
    import_session_parser = subparsers.add_parser(
        "import-session",
        help="Import a specific Claude Code session file"
    )
    import_session_parser.add_argument(
        "session_path",
        help="Path to the session .jsonl file"
    )
    import_session_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Import even if already imported"
    )

    # Main command arguments
    parser.add_argument(
        "--tools", "-t",
        nargs="+",
        choices=["claude_code", "claude", "opencode", "aider"],
        help="Specific tools to set up"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Check current setup status"
    )
    parser.add_argument(
        "--uninstall", "-u",
        action="store_true",
        help="Uninstall trace capture from all tools"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle import subcommands
    if args.command == "import-today":
        results = import_today(
            project_filter=args.project,
            verbose=verbose
        )
        imported = sum(1 for r in results if not r.skipped and not r.error)
        total_steps = sum(r.steps_imported for r in results)
        if verbose:
            print(f"\n[BashGym] Imported {imported} session(s), {total_steps} total steps")
        return 0

    if args.command == "import-recent":
        results = import_recent(
            days=args.days,
            project_filter=args.project,
            verbose=verbose
        )
        imported = sum(1 for r in results if not r.skipped and not r.error)
        total_steps = sum(r.steps_imported for r in results)
        if verbose:
            print(f"\n[BashGym] Imported {imported} session(s), {total_steps} total steps")
        return 0

    if args.command == "import-session":
        result = import_session(
            session_path=args.session_path,
            force=args.force,
            verbose=verbose
        )
        return 0 if not result.error else 1

    if args.status:
        check_setup_status(verbose=verbose)
        return 0

    if args.uninstall:
        results = uninstall_trace_capture(all_tools=True, verbose=verbose)
        return 0 if results["success"] else 1

    # Normalize tool names
    tools = None
    if args.tools:
        tools = []
        for t in args.tools:
            if t == "claude":
                tools.append("claude_code")
            else:
                tools.append(t)

    results = setup_trace_capture(tools=tools, verbose=verbose)
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
