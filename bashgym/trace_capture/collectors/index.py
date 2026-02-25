"""Cross-reference index builder.

Walks the collected records directory and builds an index mapping
session_id to all related record file paths grouped by source type.
This is the bridge between the raw collector output and the Session
Graph (Phase 2).

Index format::

    {
      "session-001": {
        "subagents": ["subagents/agent-abc.json"],
        "edits": ["edits/session-001_file.json"],
        "timestamp": "2026-02-25T10:00:00Z"
      },
      "_no_session": {
        "plans": ["plans/orphan-plan.json"],
        "timestamp": "2026-02-25T09:00:00Z"
      }
    }
"""

import json
from pathlib import Path
from typing import Dict


def build_cross_reference_index(collected_dir: Path) -> Dict[str, dict]:
    """Build a cross-reference index from collected records.

    Walks all subdirectories of *collected_dir*, reads each JSON file's
    ``session_id`` and ``source_type``, and groups file paths by session.

    The index is also written to ``collected_dir/index.json``.

    Parameters
    ----------
    collected_dir : Path
        Root of the collected records directory.

    Returns
    -------
    dict
        Mapping of session_id to record metadata.  Records with an
        empty ``session_id`` are grouped under the ``"_no_session"`` key.
    """
    collected_dir = Path(collected_dir)
    index: Dict[str, dict] = {}

    if not collected_dir.exists():
        # Create the directory and write an empty index
        collected_dir.mkdir(parents=True, exist_ok=True)
        (collected_dir / "index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8",
        )
        return index

    # Walk all source-type subdirectories
    for source_dir in sorted(collected_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        source_type = source_dir.name

        for json_file in sorted(source_dir.glob("*.json")):
            # Skip scan_state.json files used for deduplication
            if json_file.name == "scan_state.json":
                continue

            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            session_id = data.get("session_id", "")
            timestamp = data.get("timestamp", "")

            # Group empty session_id under a special key
            key = session_id if session_id else "_no_session"

            if key not in index:
                index[key] = {"timestamp": timestamp}

            # Update to the earliest timestamp
            existing_ts = index[key].get("timestamp", "")
            if timestamp and (not existing_ts or timestamp < existing_ts):
                index[key]["timestamp"] = timestamp

            # Add file path relative to collected_dir using forward slashes
            rel_path = json_file.relative_to(collected_dir).as_posix()
            if source_type not in index[key]:
                index[key][source_type] = []
            index[key][source_type].append(rel_path)

    # Write index.json to disk
    (collected_dir / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8",
    )

    return index
