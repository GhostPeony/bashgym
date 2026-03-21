"""ChatGPT conversation export importer.

Parses ChatGPT's data export format (conversations.json inside a zip)
and converts conversations into TraceSession objects.

ChatGPT export format:
- User exports from Settings -> Data Controls -> Export Data
- Receives a zip containing conversations.json
- conversations.json is an array of conversation objects
- Each conversation has a tree-structured `mapping` of messages
"""

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core import RepoInfo, TraceCapture, TraceSession, TraceStep


@dataclass
class ChatGPTImportResult:
    """Result of importing a single ChatGPT conversation."""

    session_id: str
    title: str
    steps_imported: int
    destination_file: Path | None = None
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


class ChatGPTImporter:
    """Import ChatGPT conversation exports into Bash Gym trace format."""

    def __init__(self):
        self.trace_capture = TraceCapture()
        self.imported_file = self.trace_capture.bashgym_dir / "imported_chatgpt.json"
        self._imported: set | None = None
        self._dummy_repo = RepoInfo(path="", name="chatgpt", is_git_repo=False)

    def _load_imported(self) -> set:
        if self._imported is None:
            if self.imported_file.exists():
                data = json.loads(self.imported_file.read_text())
                self._imported = set(data.get("imported_ids", []))
            else:
                self._imported = set()
        return self._imported

    def _mark_imported(self, convo_id: str):
        imported = self._load_imported()
        imported.add(convo_id)
        self.imported_file.parent.mkdir(parents=True, exist_ok=True)
        self.imported_file.write_text(json.dumps({"imported_ids": list(imported)}))

    def _walk_conversation_path(self, mapping: dict[str, Any], current_node: str) -> list[dict]:
        """Walk from current_node back to root, returning messages in chronological order."""
        path = []
        node_id = current_node
        while node_id and node_id in mapping:
            node = mapping[node_id]
            if node.get("message"):
                path.append(node["message"])
            node_id = node.get("parent")

        path.reverse()
        return path

    def parse_conversation(self, convo: dict[str, Any]) -> tuple[list[TraceStep], dict[str, Any]]:
        """Parse a single ChatGPT conversation into TraceSteps + metadata."""
        mapping = convo.get("mapping", {})
        current_node = convo.get("current_node", "")
        title = convo.get("title", "Untitled")

        if not mapping or not current_node:
            return [], {"source": "chatgpt", "title": title}

        messages = self._walk_conversation_path(mapping, current_node)
        if not messages:
            return [], {"source": "chatgpt", "title": title}

        steps = []
        models_used = set()
        user_prompts = []

        for msg in messages:
            author_role = msg.get("author", {}).get("role", "unknown")
            content = msg.get("content", {})
            content_type = content.get("content_type", "text")
            parts = content.get("parts", [])
            text = "\n".join(str(p) for p in parts if p is not None)
            create_time = msg.get("create_time")
            metadata = msg.get("metadata", {})

            timestamp = (
                datetime.fromtimestamp(create_time, tz=timezone.utc).isoformat()
                if create_time
                else datetime.now(timezone.utc).isoformat()
            )

            model_slug = metadata.get("model_slug")
            if model_slug:
                models_used.add(model_slug)

            if author_role == "user":
                user_prompts.append({"text": text[:500], "timestamp": timestamp})
                step = TraceStep.create(
                    tool_name="conversation",
                    command=text,
                    output="",
                    source_tool="chatgpt",
                    repo_info=self._dummy_repo,
                    role="user",
                    content_type=content_type,
                )
                step.success = True
                step.timestamp = timestamp
                steps.append(step)

            elif author_role == "assistant":
                if content_type == "code":
                    step = TraceStep.create(
                        tool_name="code_interpreter",
                        command=text,
                        output="",
                        source_tool="chatgpt",
                        repo_info=self._dummy_repo,
                        role="assistant",
                        content_type=content_type,
                        model=model_slug,
                    )
                else:
                    step = TraceStep.create(
                        tool_name="conversation",
                        command="",
                        output=text,
                        source_tool="chatgpt",
                        repo_info=self._dummy_repo,
                        role="assistant",
                        content_type=content_type,
                        model=model_slug,
                    )
                step.success = True
                step.timestamp = timestamp
                steps.append(step)

            elif author_role == "tool":
                tool_name = "code_interpreter" if content_type == "execution_output" else "tool"
                step = TraceStep.create(
                    tool_name=tool_name,
                    command="",
                    output=text,
                    source_tool="chatgpt",
                    repo_info=self._dummy_repo,
                    role="tool",
                    content_type=content_type,
                )
                step.success = True
                step.timestamp = timestamp
                steps.append(step)

        session_metadata = {
            "source": "chatgpt",
            "title": title,
            "create_time": convo.get("create_time"),
            "update_time": convo.get("update_time"),
            "models_used": list(models_used),
            "user_initial_prompt": user_prompts[0]["text"] if user_prompts else "",
            "all_user_prompts": user_prompts,
            "conversation_turns": len(user_prompts),
            "total_tool_calls": len([s for s in steps if s.tool_name != "conversation"]),
            "imported": True,
            "import_source": "chatgpt_export",
        }

        return steps, session_metadata

    def import_conversation(
        self, convo: dict[str, Any], force: bool = False
    ) -> ChatGPTImportResult:
        """Import a single conversation into the trace store."""
        title = convo.get("title", "Untitled")
        convo_hash = hashlib.sha256(json.dumps(convo, sort_keys=True).encode()).hexdigest()[:16]

        if not force and convo_hash in self._load_imported():
            return ChatGPTImportResult(
                session_id=convo_hash,
                title=title,
                steps_imported=0,
                skipped=True,
                skip_reason="already_imported",
            )

        steps, metadata = self.parse_conversation(convo)
        if not steps:
            return ChatGPTImportResult(
                session_id=convo_hash,
                title=title,
                steps_imported=0,
                skipped=True,
                skip_reason="empty_conversation",
            )

        session = TraceSession.from_steps(
            steps=steps,
            source_tool="chatgpt",
            **metadata,
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imported_chatgpt_{convo_hash}_{timestamp_str}.json"
        dest = self.trace_capture.traces_dir / filename
        self.trace_capture.traces_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(asdict(session), default=str, indent=2))

        self._mark_imported(convo_hash)

        return ChatGPTImportResult(
            session_id=session.session_id,
            title=title,
            steps_imported=len(steps),
            destination_file=dest,
        )

    def import_from_zip(self, zip_path: Path, force: bool = False) -> list[ChatGPTImportResult]:
        """Import all conversations from a ChatGPT export zip."""
        results = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            convo_files = [n for n in zf.namelist() if n.endswith("conversations.json")]
            if not convo_files:
                return [
                    ChatGPTImportResult(
                        session_id="",
                        title="",
                        steps_imported=0,
                        error="No conversations.json found in zip",
                    )
                ]

            for convo_file in convo_files:
                data = json.loads(zf.read(convo_file))
                if not isinstance(data, list):
                    data = [data]

                for convo in data:
                    result = self.import_conversation(convo, force=force)
                    results.append(result)

        return results

    def import_from_json(self, json_path: Path, force: bool = False) -> list[ChatGPTImportResult]:
        """Import from a raw conversations.json file."""
        data = json.loads(json_path.read_text())
        if not isinstance(data, list):
            data = [data]

        results = []
        for convo in data:
            result = self.import_conversation(convo, force=force)
            results.append(result)
        return results


# Module-level convenience functions
def import_chatgpt_sessions(zip_path=None, json_path=None, force=False, **kwargs):
    """Import ChatGPT conversations from a zip or JSON file."""
    importer = ChatGPTImporter()
    if zip_path:
        return importer.import_from_zip(Path(zip_path), force=force)
    elif json_path:
        return importer.import_from_json(Path(json_path), force=force)
    return []
