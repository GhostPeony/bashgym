import json
import pytest
import tempfile
from pathlib import Path
from bashgym.agent.memory import PeonyMemory


@pytest.fixture
def memory_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestProfile:
    def test_load_default_profile_when_none_exists(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        profile = mem.load_profile()
        assert profile["hf_username"] is None
        assert profile["preferred_base_model"] is None
        assert profile["preferred_strategy"] is None
        assert profile["projects"] == []
        assert profile["notes"] == ""

    def test_update_profile_field(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.update_profile("hf_username", "cadebrown")
        profile = mem.load_profile()
        assert profile["hf_username"] == "cadebrown"

    def test_update_profile_persists_to_disk(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.update_profile("preferred_strategy", "grpo")
        mem2 = PeonyMemory(memory_dir)
        profile = mem2.load_profile()
        assert profile["preferred_strategy"] == "grpo"

    def test_update_profile_rejects_unknown_field(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        with pytest.raises(ValueError, match="Unknown profile field"):
            mem.update_profile("nonexistent_field", "value")


class TestFacts:
    def test_remember_fact(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        fact = mem.remember_fact("project", "User is training a code assistant")
        assert fact["category"] == "project"
        assert fact["content"] == "User is training a code assistant"
        assert "id" in fact
        assert "created_at" in fact

    def test_recall_facts_by_category(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.remember_fact("project", "Building a code assistant")
        mem.remember_fact("preference", "Prefers GRPO")
        mem.remember_fact("project", "Uses Qwen base model")
        results = mem.recall_facts(category="project")
        assert len(results) == 2
        assert all(f["category"] == "project" for f in results)

    def test_recall_facts_by_keyword(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.remember_fact("project", "Building a code assistant")
        mem.remember_fact("preference", "Prefers GRPO over SFT")
        results = mem.recall_facts(keyword="GRPO")
        assert len(results) == 1
        assert "GRPO" in results[0]["content"]

    def test_recall_facts_by_category_and_keyword(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.remember_fact("project", "Code assistant with GRPO")
        mem.remember_fact("preference", "Prefers GRPO")
        results = mem.recall_facts(category="project", keyword="GRPO")
        assert len(results) == 1

    def test_forget_fact(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        fact = mem.remember_fact("project", "Temporary fact")
        mem.forget_fact(fact["id"])
        results = mem.recall_facts()
        assert len(results) == 0

    def test_forget_nonexistent_fact_raises(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        with pytest.raises(ValueError, match="Fact not found"):
            mem.forget_fact("nonexistent-id")

    def test_facts_capped_at_50(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        for i in range(55):
            mem.remember_fact("bulk", f"Fact number {i}")
        all_facts = mem.recall_facts()
        assert len(all_facts) == 50

    def test_facts_persist_to_disk(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.remember_fact("project", "Persistent fact")
        mem2 = PeonyMemory(memory_dir)
        results = mem2.recall_facts()
        assert len(results) == 1
        assert results[0]["content"] == "Persistent fact"

    def test_load_facts_with_limit(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        for i in range(10):
            mem.remember_fact("bulk", f"Fact {i}")
        results = mem.load_facts(limit=5)
        assert len(results) == 5


class TestEpisodes:
    def test_save_episode(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        episode = mem.save_episode("session_abc", "User imported 12 traces.")
        assert episode["session_id"] == "session_abc"
        assert episode["summary"] == "User imported 12 traces."
        assert "created_at" in episode

    def test_load_recent_episodes(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        for i in range(7):
            mem.save_episode(f"session_{i}", f"Summary {i}")
        episodes = mem.load_recent_episodes(limit=5)
        assert len(episodes) == 5

    def test_episodes_sorted_by_recency(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.save_episode("old", "Old summary")
        mem.save_episode("new", "New summary")
        episodes = mem.load_recent_episodes(limit=5)
        assert episodes[0]["session_id"] == "new"

    def test_episode_persists_to_disk(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.save_episode("persist_test", "Persistent episode")
        mem2 = PeonyMemory(memory_dir)
        episodes = mem2.load_recent_episodes()
        assert len(episodes) == 1
        assert episodes[0]["summary"] == "Persistent episode"


class TestMemoryPrompt:
    def test_build_memory_prompt_empty(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        prompt = mem.build_memory_prompt()
        assert "--- USER PROFILE ---" in prompt
        assert "--- KNOWN FACTS" in prompt
        assert "--- RECENT HISTORY ---" in prompt

    def test_build_memory_prompt_with_data(self, memory_dir):
        mem = PeonyMemory(memory_dir)
        mem.update_profile("hf_username", "cadebrown")
        mem.remember_fact("project", "Building a code assistant")
        mem.save_episode("s1", "Imported traces and started training.")
        prompt = mem.build_memory_prompt()
        assert "cadebrown" in prompt
        assert "Building a code assistant" in prompt
        assert "Imported traces and started training." in prompt
