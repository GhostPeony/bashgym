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
    install_gemini_cli_hooks,
    uninstall_gemini_cli_hooks,
    install_codex_hooks,
    uninstall_codex_hooks,
    install_copilot_cli_hooks,
    uninstall_copilot_cli_hooks,
)
from .importers import (
    import_today,
    import_recent,
    import_session,
    import_gemini_sessions,
    import_copilot_sessions,
    import_opencode_sessions,
)


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

        elif tool == "gemini_cli":
            success, message = install_gemini_cli_hooks()
            results["tools"]["gemini_cli"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Gemini CLI: {message}")
            if not success:
                results["success"] = False
                results["errors"].append(f"Gemini CLI: {message}")

        elif tool == "codex":
            success, message = install_codex_hooks()
            results["tools"]["codex"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Codex: {message}")
            if not success:
                results["success"] = False
                results["errors"].append(f"Codex: {message}")

        elif tool == "copilot_cli":
            success, message = install_copilot_cli_hooks()
            results["tools"]["copilot_cli"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Copilot CLI: {message}")
            if not success:
                results["success"] = False
                results["errors"].append(f"Copilot CLI: {message}")

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
        target_tools = ["claude_code", "opencode", "gemini_cli", "codex", "copilot_cli"]
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

        elif tool == "gemini_cli":
            success, message = uninstall_gemini_cli_hooks()
            results["tools"]["gemini_cli"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Gemini CLI: {message}")

        elif tool == "codex":
            success, message = uninstall_codex_hooks()
            results["tools"]["codex"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Codex: {message}")

        elif tool == "copilot_cli":
            success, message = uninstall_copilot_cli_hooks()
            results["tools"]["copilot_cli"] = {
                "success": success,
                "message": message
            }
            if verbose:
                status = "✓" if success else "✗"
                print(f"  {status} Copilot CLI: {message}")

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

  # Scan and inspect available data
  bashgym-setup scan                      # Show all available data sources
  bashgym-setup status                    # Show collection stats per source

  # Import Claude Code session history
  bashgym-setup import-today              # Import today's sessions
  bashgym-setup import-recent             # Import last 60 days (sessions)
  bashgym-setup import-recent --days 30   # Import last 30 days
  bashgym-setup import-recent -S plans    # Import plans from last 60 days
  bashgym-setup import-recent -S all      # Import all source types
  bashgym-setup import-session <path>     # Import specific session file

  # Import from other tools
  bashgym-setup import-gemini             # Import Gemini CLI sessions
  bashgym-setup import-copilot            # Import Copilot CLI sessions
  bashgym-setup import-opencode           # Import OpenCode sessions
  bashgym-setup import-all                # Import from all detected tools
        """
    )

    # Create subparsers for import commands
    subparsers = parser.add_subparsers(dest="command", help="Import commands")

    # scan subcommand
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan ~/.claude and show what data is available but not yet collected"
    )

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Show collection stats per source type"
    )

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
    import_recent_parser.add_argument(
        "--source", "-S",
        choices=["all", "sessions", "subagents", "edits", "plans", "prompts", "todos", "environments"],
        default="sessions",
        help="Data source to import (default: sessions). Use 'all' for everything."
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

    # import-gemini subcommand
    import_gemini_parser = subparsers.add_parser(
        "import-gemini",
        help="Import Gemini CLI session history"
    )
    import_gemini_parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of days to look back (default: 60)"
    )
    import_gemini_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Maximum number of sessions to import (default: 100)"
    )

    # import-copilot subcommand
    import_copilot_parser = subparsers.add_parser(
        "import-copilot",
        help="Import Copilot CLI session history"
    )
    import_copilot_parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of days to look back (default: 60)"
    )
    import_copilot_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Maximum number of sessions to import (default: 100)"
    )

    # import-opencode subcommand
    import_opencode_parser = subparsers.add_parser(
        "import-opencode",
        help="Import OpenCode session history"
    )
    import_opencode_parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of days to look back (default: 60)"
    )
    import_opencode_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Maximum number of sessions to import (default: 100)"
    )

    # import-all subcommand
    import_all_parser = subparsers.add_parser(
        "import-all",
        help="Import session history from all detected tools"
    )
    import_all_parser.add_argument(
        "--days", "-d",
        type=int,
        default=60,
        help="Number of days to look back (default: 60)"
    )
    import_all_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Maximum sessions to import per tool (default: 100)"
    )
    import_all_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-import even if already imported (overwrites existing traces)"
    )

    # Main command arguments
    parser.add_argument(
        "--tools", "-t",
        nargs="+",
        choices=["claude_code", "claude", "opencode", "gemini_cli", "gemini", "codex", "copilot_cli", "copilot", "aider"],
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

    # Handle scan and status subcommands
    if args.command == "scan":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        results = scanner.scan_all()
        if verbose:
            print("[BashGym] Data Scan Results")
            print("=" * 60)
            print(f"{'Source':<15} {'Total':<10} {'Collected':<12} {'Available':<10}")
            print("-" * 60)
            for source, result in results.items():
                print(f"{source:<15} {result.total_found:<10} {result.already_collected:<12} {result.new_available:<10}")
            total = sum(r.new_available for r in results.values())
            print("-" * 60)
            print(f"{'TOTAL':<15} {'':<10} {'':<12} {total:<10}")
        return 0

    if args.command == "status":
        from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
        scanner = ClaudeDataScanner()
        status = scanner.status()
        if verbose:
            print("[BashGym] Collection Status")
            print("=" * 60)
            print(f"{'Source':<15} {'Total':<10} {'Collected':<12} {'Available':<10}")
            print("-" * 60)
            for source, info in status.items():
                print(f"{source:<15} {info['total']:<10} {info['collected']:<12} {info['available']:<10}")
        return 0

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
        source = getattr(args, "source", "sessions")
        if source == "sessions":
            # Original behavior -- import session transcripts
            results = import_recent(
                days=args.days,
                project_filter=args.project,
                verbose=verbose
            )
            imported = sum(1 for r in results if not r.skipped and not r.error)
            total_steps = sum(r.steps_imported for r in results)
            if verbose:
                print(f"\n[BashGym] Imported {imported} session(s), {total_steps} total steps")
        else:
            # New behavior -- use ClaudeDataScanner for collector sources
            from bashgym.trace_capture.collectors.scanner import ClaudeDataScanner
            from datetime import datetime, timezone, timedelta
            scanner = ClaudeDataScanner()
            since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
            sources = None if source == "all" else [source]
            results = scanner.collect_all(
                sources=sources,
                since=since,
                project_filter=getattr(args, "project", None),
            )
            if verbose:
                for src, batch in results.items():
                    print(f"  {src}: collected={batch.collected}, skipped={batch.skipped}, errors={len(batch.errors)}")
                total = sum(r.collected for r in results.values())
                print(f"\n[BashGym] Collected {total} record(s) from {len(results)} source(s)")
        return 0

    if args.command == "import-session":
        result = import_session(
            session_path=args.session_path,
            force=args.force,
            verbose=verbose
        )
        return 0 if not result.error else 1

    if args.command == "import-gemini":
        results = import_gemini_sessions(
            days=args.days,
            limit=args.limit,
            verbose=verbose,
        )
        imported = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
        total_steps = sum(r.get("steps_imported", 0) for r in results)
        if verbose:
            print(f"\n[BashGym] Gemini: Imported {imported} session(s), {total_steps} total steps")
        return 0

    if args.command == "import-copilot":
        results = import_copilot_sessions(
            days=args.days,
            limit=args.limit,
            verbose=verbose,
        )
        imported = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
        total_steps = sum(r.get("steps_imported", 0) for r in results)
        if verbose:
            print(f"\n[BashGym] Copilot: Imported {imported} session(s), {total_steps} total steps")
        return 0

    if args.command == "import-opencode":
        results = import_opencode_sessions(
            days=args.days,
            limit=args.limit,
            verbose=verbose,
        )
        imported = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
        total_steps = sum(r.get("steps_imported", 0) for r in results)
        if verbose:
            print(f"\n[BashGym] OpenCode: Imported {imported} session(s), {total_steps} total steps")
        return 0

    if args.command == "import-all":
        if verbose:
            print("[BashGym] Importing from all detected tools...")
            print()

        grand_total_imported = 0
        grand_total_steps = 0

        # Claude Code
        if verbose:
            print("--- Claude Code ---")
        claude_results = import_recent(
            days=args.days,
            verbose=verbose,
            force=getattr(args, 'force', False),
        )
        claude_imported = sum(1 for r in claude_results if not r.skipped and not r.error)
        claude_steps = sum(r.steps_imported for r in claude_results)
        grand_total_imported += claude_imported
        grand_total_steps += claude_steps

        # Gemini CLI
        if verbose:
            print("\n--- Gemini CLI ---")
        gemini_results = import_gemini_sessions(
            days=args.days,
            limit=args.limit,
            verbose=verbose,
        )
        gemini_imported = sum(1 for r in gemini_results if not r.get("skipped") and not r.get("error"))
        gemini_steps = sum(r.get("steps_imported", 0) for r in gemini_results)
        grand_total_imported += gemini_imported
        grand_total_steps += gemini_steps

        # Copilot CLI
        if verbose:
            print("\n--- Copilot CLI ---")
        copilot_results = import_copilot_sessions(
            days=args.days,
            limit=args.limit,
            verbose=verbose,
        )
        copilot_imported = sum(1 for r in copilot_results if not r.get("skipped") and not r.get("error"))
        copilot_steps = sum(r.get("steps_imported", 0) for r in copilot_results)
        grand_total_imported += copilot_imported
        grand_total_steps += copilot_steps

        # OpenCode
        if verbose:
            print("\n--- OpenCode ---")
        opencode_results = import_opencode_sessions(
            days=args.days,
            limit=args.limit,
            verbose=verbose,
        )
        opencode_imported = sum(1 for r in opencode_results if not r.get("skipped") and not r.get("error"))
        opencode_steps = sum(r.get("steps_imported", 0) for r in opencode_results)
        grand_total_imported += opencode_imported
        grand_total_steps += opencode_steps

        if verbose:
            print(f"\n[BashGym] Total: Imported {grand_total_imported} session(s), {grand_total_steps} total steps")
            print(f"  Claude: {claude_imported}, Gemini: {gemini_imported}, "
                  f"Copilot: {copilot_imported}, OpenCode: {opencode_imported}")
        return 0

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
            elif t == "gemini":
                tools.append("gemini_cli")
            elif t == "copilot":
                tools.append("copilot_cli")
            else:
                tools.append(t)

    results = setup_trace_capture(tools=tools, verbose=verbose)
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
