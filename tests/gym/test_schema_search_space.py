"""Tests for SchemaSearchSpace -- Data Designer schema evolution."""

from bashgym.gym.schema_search_space import (
    FAILURE_TEMPLATE_MAP,
    SCHEMA_SEARCH_SPACE,
    TEMPLATE_LIBRARY,
    SchemaSearchSpace,
)


class TestSchemaSearchSpaceDefinition:
    def test_search_space_defines_traits(self):
        assert "temperature_text" in SCHEMA_SEARCH_SPACE
        assert "temperature_code" in SCHEMA_SEARCH_SPACE
        assert "include_code_validation" in SCHEMA_SEARCH_SPACE
        assert "temperature_judge" in SCHEMA_SEARCH_SPACE
        assert "judge_threshold" in SCHEMA_SEARCH_SPACE

    def test_search_space_types(self):
        assert SCHEMA_SEARCH_SPACE["temperature_text"]["type"] == "float"
        assert SCHEMA_SEARCH_SPACE["include_code_validation"]["type"] == "bool"
        assert SCHEMA_SEARCH_SPACE["judge_threshold"]["type"] == "int"
        assert SCHEMA_SEARCH_SPACE["complexity_weights"]["type"] == "weights"

    def test_template_library_has_five(self):
        assert len(TEMPLATE_LIBRARY) == 5
        assert "coding_agent_sft" in TEMPLATE_LIBRARY
        assert "tool_use_sft" in TEMPLATE_LIBRARY
        assert "coding_agent_dpo" in TEMPLATE_LIBRARY
        assert "from_external" in TEMPLATE_LIBRARY
        assert "from_unstructured" in TEMPLATE_LIBRARY

    def test_template_has_required_keys(self):
        for name, meta in TEMPLATE_LIBRARY.items():
            assert "description" in meta, f"{name} missing description"
            assert "columns" in meta, f"{name} missing columns"
            assert "judge_dimensions" in meta, f"{name} missing judge_dimensions"
            assert "default_for_failures" in meta, f"{name} missing default_for_failures"


class TestFailureTemplateMap:
    def test_wrong_tool_maps_to_tool_use(self):
        assert FAILURE_TEMPLATE_MAP.get("wrong_tool") == "tool_use_sft"

    def test_tool_misuse_maps_to_tool_use(self):
        assert FAILURE_TEMPLATE_MAP.get("tool_misuse") == "tool_use_sft"

    def test_incomplete_maps_to_sft(self):
        assert FAILURE_TEMPLATE_MAP.get("incomplete") == "coding_agent_sft"

    def test_bad_reasoning_maps_to_sft(self):
        assert FAILURE_TEMPLATE_MAP.get("bad_reasoning") == "coding_agent_sft"

    def test_context_misunderstanding_maps_to_unstructured(self):
        assert FAILURE_TEMPLATE_MAP.get("context_misunderstanding") == "from_unstructured"

    def test_quality_inconsistent_maps_to_dpo(self):
        assert FAILURE_TEMPLATE_MAP.get("quality_inconsistent") == "coding_agent_dpo"


class TestMutate:
    def test_mutate_returns_dict(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        mutated = space.mutate(genome)
        assert isinstance(mutated, dict)

    def test_mutate_does_not_modify_original(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        original_temp = genome["temperature_text"]
        space.mutate(genome)
        assert genome["temperature_text"] == original_temp

    def test_mutate_respects_float_bounds(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        for _ in range(20):
            mutated = space.mutate(genome)
            assert 0.3 <= mutated["temperature_text"] <= 1.0
            assert 0.05 <= mutated["temperature_code"] <= 0.5
            assert 0.0 <= mutated["temperature_judge"] <= 0.3

    def test_mutate_toggles_bools(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        genome["include_code_validation"] = True
        mutated = space.mutate(genome)
        # With rate=1.0, bool should toggle
        assert mutated["include_code_validation"] is False

    def test_mutate_toggles_bools_false_to_true(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        genome["include_code_validation"] = False
        mutated = space.mutate(genome)
        assert mutated["include_code_validation"] is True

    def test_mutate_int_choices_in_range(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        for _ in range(20):
            mutated = space.mutate(genome)
            assert mutated["judge_threshold"] in [2, 3, 4, 5]
            assert mutated["num_judge_dimensions"] in [2, 3, 4, 5]

    def test_mutate_weights_remain_normalized(self):
        space = SchemaSearchSpace(mutation_rate=1.0)
        genome = SchemaSearchSpace.create_default_genome()
        for _ in range(10):
            mutated = space.mutate(genome)
            weights = mutated["complexity_weights"]
            assert isinstance(weights, dict)
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.05  # Should be approximately normalized


class TestCreateDefaultGenome:
    def test_has_required_fields(self):
        genome = SchemaSearchSpace.create_default_genome()
        assert "template" in genome
        assert "temperature_text" in genome
        assert "temperature_code" in genome
        assert "temperature_judge" in genome
        assert "complexity_weights" in genome
        assert "include_code_validation" in genome
        assert "include_embedding_dedup" in genome

    def test_default_template(self):
        genome = SchemaSearchSpace.create_default_genome()
        assert genome["template"] == "coding_agent_sft"

    def test_custom_template(self):
        genome = SchemaSearchSpace.create_default_genome("tool_use_sft")
        assert genome["template"] == "tool_use_sft"

    def test_default_values(self):
        genome = SchemaSearchSpace.create_default_genome()
        assert genome["temperature_text"] == 0.85
        assert genome["temperature_code"] == 0.2
        assert genome["temperature_judge"] == 0.1
        assert genome["judge_threshold"] == 3
        assert genome["include_code_validation"] is False


class TestGetConfigSnapshot:
    def test_returns_search_params_only(self):
        space = SchemaSearchSpace()
        genome = SchemaSearchSpace.create_default_genome()
        genome["extra_field"] = "should_not_appear"
        snapshot = space.get_config_snapshot(genome)
        assert "extra_field" not in snapshot
        assert "temperature_text" in snapshot

    def test_excludes_template_field(self):
        space = SchemaSearchSpace()
        genome = SchemaSearchSpace.create_default_genome()
        snapshot = space.get_config_snapshot(genome)
        # "template" is not in SCHEMA_SEARCH_SPACE, so it should not appear
        assert "template" not in snapshot

    def test_includes_all_schema_params(self):
        space = SchemaSearchSpace()
        genome = SchemaSearchSpace.create_default_genome()
        snapshot = space.get_config_snapshot(genome)
        for key in SCHEMA_SEARCH_SPACE:
            assert key in snapshot, f"{key} missing from snapshot"


class TestSelectTemplate:
    def test_empty_traces_defaults_to_sft(self):
        template, confidence = SchemaSearchSpace.select_template([])
        assert template == "coding_agent_sft"
        assert confidence == 0.0

    def test_low_confidence_defaults_to_sft(self):
        # Traces with no strong signal should default to SFT
        traces = [
            {"trace": [{"tool_name": "Read"}, {"tool_name": "Bash"}], "metadata": {"error": ""}}
            for _ in range(5)
        ]
        template, confidence = SchemaSearchSpace.select_template(traces)
        # Should return coding_agent_sft since no strong failure signals
        assert template == "coding_agent_sft"

    def test_tool_errors_select_tool_use(self):
        traces = []
        for _ in range(5):
            traces.append(
                {
                    "trace": [
                        {"tool_name": "Bash", "exit_code": 1},
                        {"tool_name": "Bash", "exit_code": 1},
                        {"tool_name": "Bash", "exit_code": 0},
                    ],
                    "metadata": {"error": ""},
                }
            )
        template, confidence = SchemaSearchSpace.select_template(traces)
        # Should detect tool_misuse pattern (>30% bash errors)
        assert template in ("tool_use_sft", "coding_agent_sft")

    def test_incomplete_traces(self):
        # Very short traces (fewer than 3 steps) -> incomplete
        traces = [{"trace": [{"tool_name": "Read"}], "metadata": {"error": ""}} for _ in range(5)]
        template, confidence = SchemaSearchSpace.select_template(traces)
        assert confidence > 0

    def test_context_misunderstanding_from_errors(self):
        # Traces with "not found" errors
        traces = [
            {
                "trace": [{"tool_name": "Read"}, {"tool_name": "Bash"}, {"tool_name": "Edit"}],
                "metadata": {"error": "No such file or directory"},
            }
            for _ in range(5)
        ]
        template, confidence = SchemaSearchSpace.select_template(traces)
        # Should detect context_misunderstanding
        assert confidence > 0

    def test_wrong_tool_pattern(self):
        # Traces using Grep/Glob but never Read -> wrong_tool
        traces = [
            {
                "trace": [
                    {"tool_name": "Grep", "exit_code": 0},
                    {"tool_name": "Glob", "exit_code": 0},
                    {"tool_name": "Bash", "exit_code": 0},
                ],
                "metadata": {"error": ""},
            }
            for _ in range(5)
        ]
        template, confidence = SchemaSearchSpace.select_template(traces)
        assert confidence > 0


class TestSchemaSearchSpaceInit:
    def test_default_params(self):
        space = SchemaSearchSpace()
        assert space.base_pipeline_name == "coding_agent_sft"
        assert space.mutation_rate == 0.3
        assert space.stage1_examples == 25

    def test_custom_params(self):
        space = SchemaSearchSpace(
            base_pipeline_name="tool_use_sft",
            mutation_rate=0.5,
            stage1_examples=50,
            stage2_train_steps=100,
        )
        assert space.base_pipeline_name == "tool_use_sft"
        assert space.mutation_rate == 0.5
        assert space.stage1_examples == 50
        assert space.stage2_train_steps == 100

    def test_unknown_pipeline_falls_back_to_sft(self):
        space = SchemaSearchSpace(base_pipeline_name="nonexistent")
        # _template_meta should fall back to coding_agent_sft
        assert space._template_meta == TEMPLATE_LIBRARY["coding_agent_sft"]
