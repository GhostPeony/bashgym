"""Tests for the contamination-free held-out split."""

from bashgym.eval.split import contamination, make_holdout_split


def _ex(session, repo, i):
    return {"metadata": {"session_id": session, "primary_repo": {"name": repo}}, "i": i}


# 4 sessions x 3 examples each, across 2 repos.
EXAMPLES = [_ex(f"s{s}", "repoA" if s < 2 else "repoB", i) for s in range(4) for i in range(3)]


class TestHoldoutSplit:
    def test_session_split_keeps_sessions_whole(self):
        split = make_holdout_split(EXAMPLES, by="session", frac=0.25, seed=0)
        holdout_sessions = {e["metadata"]["session_id"] for e in split.holdout}
        train_sessions = {e["metadata"]["session_id"] for e in split.train}
        # No session appears in both train and holdout (no leakage).
        assert holdout_sessions.isdisjoint(train_sessions)
        assert len(split.train) + len(split.holdout) == len(EXAMPLES)

    def test_repo_split_holds_out_whole_repo(self):
        split = make_holdout_split(EXAMPLES, by="repo", frac=0.5, seed=0)
        holdout_repos = {e["metadata"]["primary_repo"]["name"] for e in split.holdout}
        train_repos = {e["metadata"]["primary_repo"]["name"] for e in split.train}
        assert holdout_repos.isdisjoint(train_repos)

    def test_manifest_and_hashes(self):
        split = make_holdout_split(EXAMPLES, by="session", frac=0.25, seed=0)
        m = split.manifest()
        assert m["by"] == "session"
        assert m["n_holdout"] == len(split.holdout)
        assert len(m["holdout_hashes"]) == len(split.holdout)

    def test_deterministic_with_seed(self):
        a = make_holdout_split(EXAMPLES, by="session", frac=0.25, seed=3)
        b = make_holdout_split(EXAMPLES, by="session", frac=0.25, seed=3)
        assert a.holdout_hashes == b.holdout_hashes


class TestContamination:
    def test_clean_split_has_no_contamination(self):
        split = make_holdout_split(EXAMPLES, by="session", frac=0.25, seed=0)
        assert contamination(split.train, split.holdout_hashes) == []

    def test_leaked_example_is_detected(self):
        split = make_holdout_split(EXAMPLES, by="session", frac=0.25, seed=0)
        leaked = split.holdout[0]
        assert contamination(split.train + [leaked], split.holdout_hashes) != []
