#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from nemo_gym.base_resources_server import BaseRunRequest


FORMAT_INSTRUCTION = (
    "\n\nOutput format: Return the final Prolog rule in either "
    "[RULE]...[/RULE] tags or a fenced ```prolog``` code block."
)


def _first_present(row: dict[str, Any], keys: list[str]) -> Optional[Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _extract_evaluation_config(row: dict[str, Any]) -> dict[str, str]:
    config = row.get("evaluation_config")
    if isinstance(config, dict):
        positive = config.get("positive_predicate")
        negative = config.get("negative_predicate")
        if positive and negative:
            return {
                "positive_predicate": str(positive),
                "negative_predicate": str(negative),
            }

    positive = _first_present(
        row,
        [
            "positive_predicate",
            "positive predicate",
            "positive_label",
            "positive label",
        ],
    )
    negative = _first_present(
        row,
        [
            "negative_predicate",
            "negative predicate",
            "negative_label",
            "negative label",
        ],
    )

    return {
        "positive_predicate": str(positive or "eastbound"),
        "negative_predicate": str(negative or "westbound"),
    }


def convert_record_to_row(record: dict[str, Any], include_format_instruction: bool = True) -> dict[str, Any]:
    prompt = _first_present(record, ["prompt", "question", "input"])
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing prompt field in record keys={sorted(record.keys())}")

    validation_program = _first_present(record, ["validation_program", "validation program"])
    if not isinstance(validation_program, str) or not validation_program.strip():
        raise ValueError(f"Missing validation program in record keys={sorted(record.keys())}")

    prompt_text = prompt.strip()
    if include_format_instruction:
        prompt_text += FORMAT_INSTRUCTION

    row = {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ]
        },
        "validation_program": validation_program,
        "evaluation_config": _extract_evaluation_config(record),
    }

    # Smoke-validate that BaseRunRequest shape is correct.
    BaseRunRequest.model_validate(row)
    return row


def write_rows_to_jsonl(rows: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def create_split_jsonl(
    dataset: Iterable[dict[str, Any]],
    output_path: Path,
    *,
    include_format_instruction: bool,
    max_rows: Optional[int] = None,
) -> int:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(dataset):
        if max_rows is not None and index >= max_rows:
            break
        rows.append(convert_record_to_row(dict(record), include_format_instruction=include_format_instruction))

    return write_rows_to_jsonl(rows, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create NeMo Gym SLR-Bench datasets from Hugging Face.")
    parser.add_argument("--repo-id", default="AIML-TUDA/SLR-Bench")
    parser.add_argument("--config-name", default="v1-All")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--output-dir", default="resources_servers/slr_bench/data")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument(
        "--no-format-instruction",
        action="store_true",
        help="Disable appending explicit output-format instructions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `datasets` to run this script: pip install datasets") from exc

    output_dir = Path(args.output_dir)
    include_format_instruction = not args.no_format_instruction

    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        hf_dataset = load_dataset(args.repo_id, args.config_name, split=split)
        output_path = output_dir / f"{split}.jsonl"
        num_rows = create_split_jsonl(
            hf_dataset,
            output_path,
            include_format_instruction=include_format_instruction,
            max_rows=args.max_rows_per_split,
        )
        print(f"Wrote {num_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
