# tests/test_skill_registry.py
import json
import pytest
import tempfile
from pathlib import Path
from bashgym.agent.skills.registry import SkillRegistry


@pytest.fixture
def skills_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "hf"
        d.mkdir()
        (d / "hf_cli.json").write_text(json.dumps({
            "name": "hf_cli",
            "description": "Hub operations",
            "trigger_keywords": ["hub", "download", "upload", "repo", "cache", "login"],
            "tools": [
                {
                    "name": "hf_download_model",
                    "description": "Download a model from HuggingFace Hub",
                    "input_schema": {
                        "type": "object",
                        "properties": {"model_id": {"type": "string"}},
                        "required": ["model_id"],
                    },
                }
            ],
            "knowledge": "The hf CLI provides hub operations."
        }))
        (d / "hf_trainer.json").write_text(json.dumps({
            "name": "hf_model_trainer",
            "description": "Fine-tune models with TRL",
            "trigger_keywords": ["train", "fine-tune", "finetune", "sft", "dpo", "grpo", "lora"],
            "tools": [
                {
                    "name": "hf_start_cloud_training",
                    "description": "Submit cloud training job",
                    "input_schema": {
                        "type": "object",
                        "properties": {"model_id": {"type": "string"}},
                        "required": ["model_id"],
                    },
                }
            ],
            "knowledge": "TRL supports SFT, DPO, GRPO training."
        }))
        yield Path(tmpdir)


class TestSkillRegistry:
    def test_loads_manifests(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        assert len(reg.skills) == 2

    def test_match_by_keyword(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        matches = reg.match("I want to download a model from the hub")
        assert len(matches) >= 1
        assert matches[0]["name"] == "hf_cli"

    def test_match_returns_top_n(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        matches = reg.match("train a model and upload to hub", top_n=2)
        assert len(matches) == 2

    def test_match_no_results(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        matches = reg.match("what is the weather today")
        assert len(matches) == 0

    def test_get_tools_for_matches(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        matches = reg.match("download a model")
        tools = reg.get_tools(matches)
        assert any(t["name"] == "hf_download_model" for t in tools)

    def test_get_knowledge_for_matches(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        matches = reg.match("train with grpo")
        knowledge = reg.get_knowledge(matches)
        assert "TRL supports SFT, DPO, GRPO" in knowledge

    def test_list_all_skills(self, skills_dir):
        reg = SkillRegistry(skills_dir)
        listing = reg.list_all()
        assert len(listing) == 2
        assert all("name" in s and "description" in s for s in listing)
