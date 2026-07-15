"""Isolated, fail-closed Git lineage for code-mutating hypotheses."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator

from bashgym.campaigns.contracts import (
    CodeLineageRecord,
    CodeLineageState,
    CodeMutationKind,
    FrozenContractModel,
    Identifier,
    canonical_hash,
    utc_now,
)

_CODE_VARIABLE_PREFIXES = (
    (("trainer.", "algorithm."), CodeMutationKind.TRAINER),
    (("gym.", "environment."), CodeMutationKind.GYM),
    (("reward.",), CodeMutationKind.REWARD),
    (("evaluator.", "verifier."), CodeMutationKind.EVALUATOR),
)


class GitLineageError(RuntimeError):
    """Stable public error without command, path, or repository details."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _safe_relative_path(raw_path: str) -> str:
    path = PurePosixPath(raw_path)
    normalized = path.as_posix()
    if (
        not raw_path
        or raw_path.startswith("/")
        or "\\" in raw_path
        or normalized != raw_path
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 for character in raw_path)
    ):
        raise ValueError("source mutation paths must be safe repository-relative paths")
    return normalized


class ApprovedSourceRepositoryProfile(FrozenContractModel):
    """Installation-owned repository path and allowed code-mutation scopes."""

    schema_version: Literal["campaign_source_repository_profile.v1"] = (
        "campaign_source_repository_profile.v1"
    )
    profile_id: Identifier
    repository_path: Path
    allowed_mutation_paths: dict[CodeMutationKind, tuple[str, ...]] = Field(min_length=1)

    @field_validator("repository_path")
    @classmethod
    def validate_repository_path(cls, value: Path) -> Path:
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError("source repository path must be absolute")
        return expanded

    @field_validator("allowed_mutation_paths")
    @classmethod
    def validate_allowed_paths(
        cls, value: dict[CodeMutationKind, tuple[str, ...]]
    ) -> dict[CodeMutationKind, tuple[str, ...]]:
        normalized: dict[CodeMutationKind, tuple[str, ...]] = {}
        for kind, paths in value.items():
            clean = tuple(sorted({_safe_relative_path(path) for path in paths}))
            if not clean:
                raise ValueError("each source mutation kind requires an approved path")
            normalized[kind] = clean
        return dict(sorted(normalized.items(), key=lambda item: item[0].value))

    @property
    def profile_digest(self) -> str:
        payload = self.model_dump(mode="json")
        payload["repository_path"] = str(self.repository_path.expanduser().resolve())
        return canonical_hash(payload)


@dataclass(frozen=True)
class CodeLineageWorkspaceReceipt:
    """Prepared durable record plus its private local worktree location."""

    record: CodeLineageRecord
    worktree_path: Path


def code_mutation_kind_for_variable(variable: str) -> CodeMutationKind | None:
    """Classify variables that require Git lineage; recipe scalars return ``None``."""

    normalized = variable.strip().lower()
    for prefixes, kind in _CODE_VARIABLE_PREFIXES:
        if normalized.startswith(prefixes):
            return kind
    return None


class GitHypothesisLineageManager:
    """Prepare one retained branch/worktree and capture one scoped commit."""

    def __init__(self, worktree_root: Path) -> None:
        self.worktree_root = worktree_root.expanduser()

    @staticmethod
    def _invoke(
        cwd: Path,
        *args: str,
        allowed_codes: tuple[int, ...] = (0,),
        binary: bool = False,
        disable_hooks: Path | None = None,
    ) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
        command = ["git"]
        if disable_hooks is not None:
            command.extend(("-c", f"core.hooksPath={disable_hooks}"))
        command.extend(args)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        }
        environment["GIT_TERMINAL_PROMPT"] = "0"
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=not binary,
                timeout=30,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitLineageError("campaign_git_lineage_command_failed") from exc
        if completed.returncode not in allowed_codes:
            raise GitLineageError("campaign_git_lineage_command_failed")
        return completed

    @classmethod
    def _text(cls, cwd: Path, *args: str) -> str:
        result = cls._invoke(cwd, *args)
        assert isinstance(result.stdout, str)
        return result.stdout.strip()

    def _roots(self) -> tuple[Path, Path]:
        if self.worktree_root.is_symlink():
            raise GitLineageError("campaign_git_lineage_worktree_root_unsafe")
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        root = self.worktree_root.resolve()
        hooks = root / ".disabled-hooks"
        hooks.mkdir(exist_ok=True)
        if hooks.is_symlink():
            raise GitLineageError("campaign_git_lineage_worktree_root_unsafe")
        return root, hooks

    def _repository(self, profile: ApprovedSourceRepositoryProfile) -> Path:
        supplied = profile.repository_path.expanduser()
        if not supplied.is_dir() or supplied.is_symlink():
            raise GitLineageError("campaign_git_lineage_repository_unavailable")
        repository = supplied.resolve()
        if self._text(repository, "rev-parse", "--is-inside-work-tree") != "true":
            raise GitLineageError("campaign_git_lineage_repository_unavailable")
        if self._text(repository, "rev-parse", "--show-prefix"):
            raise GitLineageError("campaign_git_lineage_repository_root_required")
        return repository

    def verify_profile(self, profile: ApprovedSourceRepositoryProfile) -> None:
        """Verify the private repository and every approved scope without exposing paths."""

        repository = self._repository(profile)
        for approved_paths in profile.allowed_mutation_paths.values():
            for relative_path in approved_paths:
                target = repository.joinpath(*PurePosixPath(relative_path).parts)
                try:
                    resolved = target.resolve(strict=True)
                    resolved.relative_to(repository)
                except (OSError, ValueError) as exc:
                    raise GitLineageError(
                        "campaign_git_lineage_approved_path_unavailable"
                    ) from exc
                if target.is_symlink():
                    raise GitLineageError(
                        "campaign_git_lineage_approved_path_unsafe"
                    )

    @staticmethod
    def _validate_profile_binding(
        profile: ApprovedSourceRepositoryProfile, record: CodeLineageRecord
    ) -> None:
        if profile.profile_id != record.source_repository_profile_id:
            raise GitLineageError("campaign_git_lineage_profile_mismatch")
        if record.mutation_kind not in profile.allowed_mutation_paths:
            raise GitLineageError("campaign_git_lineage_mutation_kind_not_approved")

    @staticmethod
    def _branch_name(record: CodeLineageRecord, base_commit: str) -> str:
        slug = re.sub(r"[^a-z0-9._-]+", "-", record.proposal_id.lower()).strip("-.")
        while ".." in slug:
            slug = slug.replace("..", "-")
        slug = slug[:64] or "proposal"
        suffix = canonical_hash(
            [
                record.lineage_id,
                record.workspace_id,
                record.campaign_id,
                record.proposal_id,
                record.mutation_kind.value,
                record.source_repository_profile_id,
                base_commit,
            ]
        )[:16]
        return f"bashgym/autoresearch/{slug}-{suffix}"

    def _worktree_path(self, record: CodeLineageRecord) -> Path:
        root, _hooks = self._roots()
        directory = canonical_hash(
            [record.workspace_id, record.campaign_id, record.proposal_id, record.lineage_id]
        )[:24]
        path = (root / directory).resolve()
        if path.parent != root:
            raise GitLineageError("campaign_git_lineage_worktree_path_unsafe")
        return path

    @classmethod
    def _branch_commit(cls, repository: Path, branch_name: str) -> str | None:
        result = cls._invoke(
            repository,
            "for-each-ref",
            "--format=%(objectname)",
            f"refs/heads/{branch_name}",
        )
        assert isinstance(result.stdout, str)
        value = result.stdout.strip()
        return value or None

    @classmethod
    def _verify_prepared_worktree(
        cls, worktree: Path, branch_name: str, expected_head: str
    ) -> None:
        if not worktree.is_dir() or worktree.is_symlink():
            raise GitLineageError("campaign_git_lineage_worktree_unavailable")
        head = cls._text(worktree, "rev-parse", "HEAD")
        branch = cls._text(worktree, "symbolic-ref", "--short", "HEAD")
        if head != expected_head or branch != branch_name:
            raise GitLineageError("campaign_git_lineage_worktree_identity_mismatch")

    def prepare(
        self,
        profile: ApprovedSourceRepositoryProfile,
        record: CodeLineageRecord,
        *,
        base_ref: str = "HEAD",
    ) -> CodeLineageWorkspaceReceipt:
        """Create or recover the isolated hypothesis branch without merging it."""

        self._validate_profile_binding(profile, record)
        if record.state == CodeLineageState.CAPTURED:
            self._verify_captured(profile, record)
            return CodeLineageWorkspaceReceipt(record, self._worktree_path(record))
        repository = self._repository(profile)
        worktree = self._worktree_path(record)
        _root, hooks = self._roots()

        if record.state == CodeLineageState.PREPARED:
            assert record.base_commit is not None and record.branch_name is not None
            if self._branch_commit(repository, record.branch_name) != record.base_commit:
                raise GitLineageError("campaign_git_lineage_branch_identity_mismatch")
            self._verify_prepared_worktree(worktree, record.branch_name, record.base_commit)
            return CodeLineageWorkspaceReceipt(record, worktree)
        if record.state != CodeLineageState.REQUIRED:
            raise GitLineageError("campaign_git_lineage_state_invalid")

        base_commit = self._text(repository, "rev-parse", "--verify", base_ref)
        if self._text(repository, "cat-file", "-t", base_commit) != "commit":
            raise GitLineageError("campaign_git_lineage_base_not_commit")
        branch_name = self._branch_name(record, base_commit)
        branch_commit = self._branch_commit(repository, branch_name)
        if worktree.exists() and not worktree.is_dir():
            raise GitLineageError("campaign_git_lineage_worktree_path_unsafe")
        if branch_commit is None:
            if worktree.exists():
                raise GitLineageError("campaign_git_lineage_worktree_identity_mismatch")
            self._invoke(
                repository,
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree),
                base_commit,
                disable_hooks=hooks,
            )
        else:
            if branch_commit != base_commit:
                raise GitLineageError("campaign_git_lineage_branch_identity_mismatch")
            if not worktree.exists():
                self._invoke(
                    repository,
                    "worktree",
                    "add",
                    str(worktree),
                    branch_name,
                    disable_hooks=hooks,
                )
        self._verify_prepared_worktree(worktree, branch_name, base_commit)
        updated_at = max(utc_now(), record.created_at)
        prepared = CodeLineageRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "state": CodeLineageState.PREPARED,
                "base_commit": base_commit,
                "branch_name": branch_name,
                "updated_at": updated_at,
            }
        )
        return CodeLineageWorkspaceReceipt(prepared, worktree)

    @classmethod
    def _nul_paths(cls, cwd: Path, *args: str) -> tuple[str, ...]:
        result = cls._invoke(cwd, *args, binary=True)
        assert isinstance(result.stdout, bytes)
        try:
            paths = {
                item.decode("utf-8")
                for item in result.stdout.split(b"\0")
                if item
            }
        except UnicodeDecodeError as exc:
            raise GitLineageError("campaign_git_lineage_path_encoding_invalid") from exc
        try:
            return tuple(sorted(_safe_relative_path(path) for path in paths))
        except ValueError as exc:
            raise GitLineageError("campaign_git_lineage_path_unsafe") from exc

    @staticmethod
    def _path_is_approved(path: str, approved_roots: tuple[str, ...]) -> bool:
        return any(path == root or path.startswith(f"{root}/") for root in approved_roots)

    @classmethod
    def _reject_special_entries(cls, worktree: Path, base_commit: str, path: str) -> None:
        target = worktree.joinpath(*PurePosixPath(path).parts)
        if target.is_symlink():
            raise GitLineageError("campaign_git_lineage_special_entry_rejected")
        for args in (
            ("ls-files", "--stage", "--", path),
            ("ls-tree", base_commit, "--", path),
        ):
            output = cls._text(worktree, *args)
            if output.startswith(("120000 ", "160000 ")):
                raise GitLineageError("campaign_git_lineage_special_entry_rejected")

    def capture(
        self,
        profile: ApprovedSourceRepositoryProfile,
        record: CodeLineageRecord,
    ) -> CodeLineageRecord:
        """Commit exactly one approved patch and return immutable Git evidence."""

        self._validate_profile_binding(profile, record)
        if record.state == CodeLineageState.CAPTURED:
            self._verify_captured(profile, record)
            return record
        if record.state != CodeLineageState.PREPARED:
            raise GitLineageError("campaign_git_lineage_not_prepared")
        assert record.base_commit is not None and record.branch_name is not None
        repository = self._repository(profile)
        worktree = self._worktree_path(record)
        _root, hooks = self._roots()
        branch_commit = self._branch_commit(repository, record.branch_name)
        if branch_commit is None:
            raise GitLineageError("campaign_git_lineage_branch_identity_mismatch")
        if branch_commit != record.base_commit:
            if not worktree.is_dir() or worktree.is_symlink():
                raise GitLineageError("campaign_git_lineage_worktree_unavailable")
            if (
                self._text(worktree, "symbolic-ref", "--short", "HEAD")
                != record.branch_name
                or self._text(worktree, "rev-parse", "HEAD") != branch_commit
            ):
                raise GitLineageError(
                    "campaign_git_lineage_worktree_identity_mismatch"
                )
            recovered = self._captured_record_from_commit(
                profile, record, worktree, branch_commit
            )
            self._verify_captured(profile, recovered)
            return recovered
        self._verify_prepared_worktree(worktree, record.branch_name, record.base_commit)

        conflicts = self._nul_paths(
            worktree,
            "diff",
            "--no-ext-diff",
            "--name-only",
            "-z",
            "--diff-filter=U",
        )
        if conflicts:
            raise GitLineageError("campaign_git_lineage_conflict")
        changed_paths = tuple(
            sorted(
                set(
                    self._nul_paths(
                        worktree,
                        "diff",
                        "--no-ext-diff",
                        "--name-only",
                        "-z",
                        "--no-renames",
                    )
                )
                | set(
                    self._nul_paths(
                        worktree,
                        "diff",
                        "--no-ext-diff",
                        "--cached",
                        "--name-only",
                        "-z",
                        "--no-renames",
                    )
                )
                | set(
                    self._nul_paths(
                        worktree,
                        "ls-files",
                        "--others",
                        "--exclude-standard",
                        "-z",
                    )
                )
            )
        )
        if not changed_paths:
            raise GitLineageError("campaign_git_lineage_empty_change")
        approved_roots = profile.allowed_mutation_paths[record.mutation_kind]
        if any(
            not self._path_is_approved(path, approved_roots) for path in changed_paths
        ):
            raise GitLineageError("campaign_git_lineage_path_not_approved")
        for path in changed_paths:
            self._reject_special_entries(worktree, record.base_commit, path)

        self._invoke(worktree, "add", "-A", "--", *changed_paths, disable_hooks=hooks)
        staged_paths = self._nul_paths(
            worktree,
            "diff",
            "--no-ext-diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
        )
        if staged_paths != changed_paths:
            raise GitLineageError("campaign_git_lineage_staged_paths_mismatch")
        self._invoke(
            worktree,
            "-c",
            "user.name=BashGym AutoResearch",
            "-c",
            "user.email=autoresearch@localhost",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "--no-verify",
            "-m",
            f"autoresearch: capture {record.proposal_id}",
            "-m",
            f"BashGym-Lineage: {record.lineage_id}",
            disable_hooks=hooks,
        )
        commit_sha = self._text(worktree, "rev-parse", "HEAD")
        captured = self._captured_record_from_commit(
            profile, record, worktree, commit_sha
        )
        self._verify_captured(profile, captured)
        return captured

    def _captured_record_from_commit(
        self,
        profile: ApprovedSourceRepositoryProfile,
        record: CodeLineageRecord,
        worktree: Path,
        commit_sha: str,
    ) -> CodeLineageRecord:
        assert record.base_commit is not None
        self._verify_single_child_commit(worktree, record.base_commit, commit_sha)
        message = self._text(worktree, "show", "-s", "--format=%B", commit_sha)
        if f"BashGym-Lineage: {record.lineage_id}" not in message.splitlines():
            raise GitLineageError("campaign_git_lineage_commit_identity_mismatch")
        committed_paths = self._nul_paths(
            worktree,
            "diff-tree",
            "--no-ext-diff",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "--no-renames",
            commit_sha,
        )
        if not committed_paths:
            raise GitLineageError("campaign_git_lineage_empty_change")
        approved_roots = profile.allowed_mutation_paths[record.mutation_kind]
        if any(
            not self._path_is_approved(path, approved_roots)
            for path in committed_paths
        ):
            raise GitLineageError("campaign_git_lineage_path_not_approved")
        patch = self._invoke(
            worktree,
            "diff",
            "--no-ext-diff",
            "--binary",
            record.base_commit,
            commit_sha,
            binary=True,
        )
        assert isinstance(patch.stdout, bytes)
        commit_time = datetime.fromisoformat(
            self._text(worktree, "show", "-s", "--format=%cI", commit_sha)
        )
        captured_at = max(commit_time, record.updated_at, record.created_at)
        return CodeLineageRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "state": CodeLineageState.CAPTURED,
                "commit_sha": commit_sha,
                "changed_paths": committed_paths,
                "patch_sha256": hashlib.sha256(patch.stdout).hexdigest(),
                "captured_at": captured_at,
                "updated_at": captured_at,
            }
        )

    @classmethod
    def _verify_single_child_commit(
        cls, repository: Path, base_commit: str, commit_sha: str
    ) -> None:
        parents = cls._text(
            repository, "rev-list", "--parents", "-n", "1", commit_sha
        ).split()
        if parents != [commit_sha, base_commit]:
            raise GitLineageError("campaign_git_lineage_commit_parent_invalid")

    def _verify_captured(
        self, profile: ApprovedSourceRepositoryProfile, record: CodeLineageRecord
    ) -> None:
        if record.state != CodeLineageState.CAPTURED:
            raise GitLineageError("campaign_git_lineage_not_captured")
        assert (
            record.base_commit is not None
            and record.branch_name is not None
            and record.commit_sha is not None
            and record.patch_sha256 is not None
        )
        repository = self._repository(profile)
        if self._branch_commit(repository, record.branch_name) != record.commit_sha:
            raise GitLineageError("campaign_git_lineage_branch_identity_mismatch")
        self._verify_single_child_commit(
            repository, record.base_commit, record.commit_sha
        )
        message = self._text(
            repository, "show", "-s", "--format=%B", record.commit_sha
        )
        if f"BashGym-Lineage: {record.lineage_id}" not in message.splitlines():
            raise GitLineageError("campaign_git_lineage_commit_identity_mismatch")
        paths = self._nul_paths(
            repository,
            "diff-tree",
            "--no-ext-diff",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            "--no-renames",
            record.commit_sha,
        )
        if paths != record.changed_paths:
            raise GitLineageError("campaign_git_lineage_commit_paths_mismatch")
        approved_roots = profile.allowed_mutation_paths[record.mutation_kind]
        if any(not self._path_is_approved(path, approved_roots) for path in paths):
            raise GitLineageError("campaign_git_lineage_path_not_approved")
        for path in paths:
            output = self._text(repository, "ls-tree", record.commit_sha, "--", path)
            if output.startswith(("120000 ", "160000 ")):
                raise GitLineageError("campaign_git_lineage_special_entry_rejected")
        patch = self._invoke(
            repository,
            "diff",
            "--no-ext-diff",
            "--binary",
            record.base_commit,
            record.commit_sha,
            binary=True,
        )
        assert isinstance(patch.stdout, bytes)
        if hashlib.sha256(patch.stdout).hexdigest() != record.patch_sha256:
            raise GitLineageError("campaign_git_lineage_patch_digest_mismatch")


__all__ = [
    "ApprovedSourceRepositoryProfile",
    "CodeLineageWorkspaceReceipt",
    "GitHypothesisLineageManager",
    "GitLineageError",
    "code_mutation_kind_for_variable",
]
