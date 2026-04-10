# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.base_resources_server import BaseRunRequest
from resources_servers.slr_bench.scripts.create_dataset import (
    convert_record_to_row,
    create_split_jsonl,
)


class TestCreateDataset:
    def test_convert_record_to_row_includes_required_fields(self) -> None:
        record = {
            "prompt": "Infer a rule for eastbound trains.",
            "validation program": "eastbound(train1).\nwestbound(train2).",
            "evaluation_config": {
                "positive_predicate": "eastbound",
                "negative_predicate": "westbound",
            },
        }

        row = convert_record_to_row(record)

        BaseRunRequest.model_validate(row)
        assert row["validation_program"] == "eastbound(train1).\nwestbound(train2)."
        assert row["evaluation_config"] == {
            "positive_predicate": "eastbound",
            "negative_predicate": "westbound",
        }
        assert "[RULE]" in row["responses_create_params"]["input"][0]["content"]

    def test_convert_record_to_row_supports_variant_fields(self) -> None:
        record = {
            "question": "Infer a train classifier.",
            "validation_program": "pos(train1).\nneg(train2).",
            "positive predicate": "foo",
            "negative predicate": "bar",
        }

        row = convert_record_to_row(record, include_format_instruction=False)

        BaseRunRequest.model_validate(row)
        assert row["evaluation_config"] == {
            "positive_predicate": "foo",
            "negative_predicate": "bar",
        }
        assert row["responses_create_params"]["input"][0]["content"] == "Infer a train classifier."

    def test_create_split_jsonl_smoke(self, tmp_path) -> None:
        dataset = [
            {
                "prompt": "Task A",
                "validation program": "eastbound(train1).",
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            },
            {
                "prompt": "Task B",
                "validation program": "eastbound(train3).",
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            },
        ]

        output_path = tmp_path / "train.jsonl"
        count = create_split_jsonl(dataset, output_path, include_format_instruction=True)

        assert count == 2
        with output_path.open() as handle:
            lines = [json.loads(line) for line in handle]

        assert len(lines) == 2
        for row in lines:
            BaseRunRequest.model_validate(row)
            assert "validation_program" in row
            assert "evaluation_config" in row
            assert row["evaluation_config"]["positive_predicate"]
            assert row["evaluation_config"]["negative_predicate"]

    def test_convert_record_to_row_raises_on_missing_validation_program(self) -> None:
        with pytest.raises(ValueError, match="Missing validation program"):
            convert_record_to_row({"prompt": "Task without validator"})
