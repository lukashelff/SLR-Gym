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

import logging
import os
import re
import subprocess
import tempfile
from typing import Optional


logger = logging.getLogger(__name__)


def _replace_label_predicates(program: str, positive_predicate: str, negative_predicate: str) -> str:
    program = re.sub(rf"\b{re.escape(positive_predicate)}\b", "pos", program)
    program = re.sub(rf"\b{re.escape(negative_predicate)}\b", "neg", program)
    return program


def _extract_arity(validation_program: str, positive_predicate: str, negative_predicate: str) -> int:
    pos_examples = re.findall(rf"{re.escape(positive_predicate)}\(([^)]*)\)", validation_program)
    neg_examples = re.findall(rf"{re.escape(negative_predicate)}\(([^)]*)\)", validation_program)

    sample = pos_examples[0] if pos_examples else (neg_examples[0] if neg_examples else "")
    if not sample.strip():
        return 1

    return sample.count(",") + 1


def _prepare_validation_program(
    validation_program: str,
    *,
    positive_predicate: str,
    negative_predicate: str,
    arity: int,
    isomorphic: bool,
) -> str:
    program = _replace_label_predicates(validation_program, positive_predicate, negative_predicate)

    if isomorphic:
        program = program.replace("(train", "(mytrain")
        program = program.replace("(car", "(mycar").replace(", car", ", mycar")

        # Avoid existence_error for shortcuts like: eastbound(T) :- \+ westbound(T).
        program = f":- dynamic {negative_predicate}/{arity}.\n" + program
        return program

    vars_str = ", ".join(f"X{i}" for i in range(1, arity + 1))
    bridge = f"\n{negative_predicate}({vars_str}) :- neg({vars_str}).\n"
    return program + bridge


def _build_symbolic_judge(
    *,
    positive_predicate: str,
    arity: int,
) -> str:
    vars_str = ", ".join(f"X{i}" for i in range(1, arity + 1))
    return f"""
check({vars_str}) :- pos({vars_str}), {positive_predicate}({vars_str}).
check({vars_str}) :- neg({vars_str}), \\+ {positive_predicate}({vars_str}).

check_count(Count) :-
    (setof(({vars_str}), ((pos({vars_str}); neg({vars_str})), check({vars_str})), CorrectExamples) ->
        length(CorrectExamples, Count)
    ;
        Count = 0
    ).
"""


def _parse_check_count(stdout: str) -> Optional[int]:
    text = stdout.strip()
    if not text:
        return None

    last_line = text.splitlines()[-1].strip()
    try:
        return int(last_line)
    except ValueError:
        return None


def evaluate_prediction(
    prediction: str,
    validation_program: str,
    eval_config: dict,
    timeout: int = 5,
    isomorphic: bool = True,
) -> dict[str, object]:
    """Evaluate a predicted Prolog rule against a validation program."""

    if not isinstance(prediction, str) or not prediction.strip():
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": "Empty prediction.",
        }

    positive_predicate = eval_config.get("positive_predicate", "eastbound")
    negative_predicate = eval_config.get("negative_predicate", "westbound")

    if not re.search(rf"\b{re.escape(positive_predicate)}\s*\(", prediction):
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Prediction does not contain target predicate '{positive_predicate}'.",
        }

    arity = _extract_arity(validation_program, positive_predicate, negative_predicate)
    prepared_program = _prepare_validation_program(
        validation_program,
        positive_predicate=positive_predicate,
        negative_predicate=negative_predicate,
        arity=arity,
        isomorphic=isomorphic,
    )

    num_examples = prepared_program.count("pos(") + prepared_program.count("neg(")
    if num_examples <= 0:
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": "No positive/negative examples found in validation program.",
        }

    symbolic_judge = _build_symbolic_judge(
        positive_predicate=positive_predicate,
        arity=arity,
    )
    full_program = (
        "\n".join(sorted(prepared_program.splitlines())) + "\n\n" + symbolic_judge + "\n" + prediction + "\n"
    )

    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pl", delete=False, encoding="utf-8") as handle:
            handle.write(full_program)
            temp_file = handle.name

        cmd = ["swipl", "-q", "-s", temp_file, "-g", "check_count(Count), writeln(Count)", "-t", "halt"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            return {
                "is_correct": False,
                "partial_score": 0.0,
                "syntax_valid": False,
                "error": result.stderr.strip() or "Prolog evaluation failed.",
            }

        correct_count = _parse_check_count(result.stdout)
        if correct_count is None:
            return {
                "is_correct": False,
                "partial_score": 0.0,
                "syntax_valid": False,
                "error": "Could not parse evaluator output.",
            }

        partial_score = correct_count / num_examples
        return {
            "is_correct": partial_score == 1.0,
            "partial_score": partial_score,
            "syntax_valid": True,
            "error": result.stderr.strip() or None,
        }

    except subprocess.TimeoutExpired:
        logger.warning("[SLRBenchVerifier] Evaluation timed out after %ss.", timeout)
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": f"Evaluation timed out after {timeout}s.",
        }
    except FileNotFoundError:
        logger.warning("[SLRBenchVerifier] Could not find `swipl` in PATH.")
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": "`swipl` not found in PATH.",
        }
    except Exception as exc:
        logger.warning("[SLRBenchVerifier] Unexpected evaluator error: %s", exc, exc_info=True)
        return {
            "is_correct": False,
            "partial_score": 0.0,
            "syntax_valid": False,
            "error": str(exc),
        }
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
