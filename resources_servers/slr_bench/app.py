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

import logging
import math
import re
from typing import Literal, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


try:
    from resources_servers.slr_bench.slr_verifier import evaluate_prediction
except ModuleNotFoundError:  # pragma: no cover
    from slr_verifier import evaluate_prediction


logger = logging.getLogger(__name__)


class SLRBenchResourcesServerConfig(BaseResourcesServerConfig):
    slr_reward: Literal["isomorphic", "base"] = "isomorphic"
    slr_reward_function: Literal["shaped", "partial"] = "shaped"
    evaluation_timeout_sec: int = 5


class SLRBenchRunRequest(BaseRunRequest):
    validation_program: Optional[str] = None
    evaluation_config: Optional[dict[str, str]] = None


class SLRBenchVerifyRequest(SLRBenchRunRequest, BaseVerifyRequest):
    pass


class SLRBenchVerifyResponse(BaseVerifyResponse):
    slr_bench_isomorphic: float = 0.0
    slr_bench_base: float = 0.0
    slr_bench_format: float = 0.0
    slr_reward_hacking: float = 0.0
    slr_bench_solved: float = 0.0
    extracted_rule: Optional[str] = None


_RULE_TAG_PATTERN = re.compile(r"\[RULE\]\s*(.*?)\s*\[\s*\\?/RULE\s*\]", re.IGNORECASE | re.DOTALL)
_PROLOG_BLOCK_PATTERN = re.compile(r"```prolog\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_CODE_BLOCK_PATTERN = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*(.*?)```", re.DOTALL)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    """Extract assistant output text from a NeMo Gym response."""
    texts: list[str] = []
    for output_item in body.response.output:
        if getattr(output_item, "type", None) != "message" or getattr(output_item, "role", None) != "assistant":
            continue

        content = getattr(output_item, "content", None)
        if isinstance(content, list):
            for content_item in content:
                if getattr(content_item, "type", None) != "output_text":
                    continue
                text = getattr(content_item, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)

    return "\n".join(texts).strip()


def _remove_thinking_sections(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _extract_rule_from_code_blocks(prediction: str) -> tuple[Optional[str], bool]:
    """Extract a Prolog rule from [RULE] tags or fenced code blocks.

    Returns:
        (rule_text, format_ok)
        - format_ok=True only when [RULE]...[/RULE] tags are used.
    """

    if not prediction or not prediction.strip():
        return None, False

    cleaned = _remove_thinking_sections(prediction)
    if not cleaned:
        return None, False

    matches = _RULE_TAG_PATTERN.findall(cleaned)
    for match in reversed(matches):
        rule = match.strip()
        if rule:
            return rule, True

    matches = _PROLOG_BLOCK_PATTERN.findall(cleaned)
    for match in reversed(matches):
        rule = match.strip()
        if rule:
            return rule, False

    matches = _CODE_BLOCK_PATTERN.findall(cleaned)
    for match in reversed(matches):
        rule = match.strip()
        if rule:
            return rule, False

    return None, False


class SLRBenchResourcesServer(SimpleResourcesServer):
    config: SLRBenchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    @staticmethod
    def get_rule_simplicity_bonus(model_response: str) -> float:
        """Compute a gentle rule simplicity bonus in [0, 1]."""
        if not model_response or not model_response.strip():
            return 0.0

        text = model_response.strip()
        rule_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*(?::-\s*[^.]+)?\."
        rules = re.findall(rule_pattern, text)

        if not rules:
            return 0.8

        num_rules = len(rules)
        total_literals = 0
        total_length = sum(len(rule) for rule in rules)

        for rule in rules:
            if ":-" in rule:
                _, body = rule.split(":-", 1)
                literals = [lit.strip() for lit in body.split(",") if lit.strip()]
                total_literals += len(literals)
            else:
                total_literals += 1

        literal_score = math.exp(-0.035 * max(0, total_literals - 6))
        rule_count_score = math.exp(-0.08 * max(0, num_rules - 2))
        length_score = math.exp(-0.001 * max(0, total_length - 120))

        simplicity = 0.6 * literal_score + 0.25 * rule_count_score + 0.15 * length_score
        return max(0.0, min(1.0, simplicity))

    def compute_reward(
        self,
        *,
        accuracy: float,
        partial_score: float,
        syntax_score: float,
        rule_simplicity_bonus: float,
    ) -> float:
        """Compute reward in [0, 1] for ILP rule induction."""
        if self.config.slr_reward_function == "partial":
            return max(0.0, min(1.0, float(partial_score)))

        if accuracy == 1.0:
            return 0.95 + 0.05 * rule_simplicity_bonus

        partial_gate = 0.5
        if partial_score < partial_gate:
            return 0.0

        k = 4
        base = float(partial_score) ** k

        # Keep syntax_score for API compatibility; it is intentionally unused in shaping.
        del syntax_score

        simplicity_mod = 0.9 + 0.1 * rule_simplicity_bonus
        return min(0.9, base * simplicity_mod)

    def _build_zero_response(
        self, body: SLRBenchVerifyRequest, *, extracted_rule: Optional[str] = None
    ) -> SLRBenchVerifyResponse:
        return SLRBenchVerifyResponse(
            **body.model_dump(),
            reward=0.0,
            slr_bench_isomorphic=0.0,
            slr_bench_base=0.0,
            slr_bench_format=0.0,
            slr_reward_hacking=0.0,
            slr_bench_solved=0.0,
            extracted_rule=extracted_rule,
        )

    async def verify(self, body: SLRBenchVerifyRequest) -> SLRBenchVerifyResponse:
        model_output = _extract_last_assistant_text(body)
        if not model_output:
            return self._build_zero_response(body)

        if not body.validation_program or not isinstance(body.validation_program, str):
            return self._build_zero_response(body)

        if not isinstance(body.evaluation_config, dict):
            return self._build_zero_response(body)

        positive_predicate = body.evaluation_config.get("positive_predicate")
        negative_predicate = body.evaluation_config.get("negative_predicate")
        if not positive_predicate or not negative_predicate:
            return self._build_zero_response(body)

        rule, format_ok = _extract_rule_from_code_blocks(model_output)
        if rule is None:
            return self._build_zero_response(body)

        rule_simplicity_bonus = self.get_rule_simplicity_bonus(rule)
        scores = {
            "slr_bench_isomorphic": 0.0,
            "slr_bench_base": 0.0,
            "slr_bench_format": 1.0 if format_ok else 0.0,
            "slr_reward_hacking": 0.0,
            "slr_bench_solved": 0.0,
        }

        for judge_name, isomorphic in (("isomorphic", True), ("base", False)):
            try:
                result = evaluate_prediction(
                    prediction=rule,
                    validation_program=body.validation_program,
                    eval_config=body.evaluation_config,
                    timeout=self.config.evaluation_timeout_sec,
                    isomorphic=isomorphic,
                )
            except Exception as exc:
                logger.warning("[SLRBenchVerifier] %s metric failed: %s", judge_name, exc, exc_info=True)
                continue

            if not isinstance(result, dict):
                logger.warning("[SLRBenchVerifier] %s returned non-dict result: %r", judge_name, result)
                continue

            try:
                accuracy = 1.0 if result.get("is_correct") else 0.0
                partial_score = float(result.get("partial_score", 0.0))
                syntax_score = 1.0 if result.get("syntax_valid") else 0.0
            except (TypeError, ValueError) as exc:
                logger.warning("[SLRBenchVerifier] %s invalid result %r: %s", judge_name, result, exc)
                continue

            score = self.compute_reward(
                accuracy=accuracy,
                partial_score=partial_score,
                syntax_score=syntax_score,
                rule_simplicity_bonus=rule_simplicity_bonus,
            )
            scores[f"slr_bench_{judge_name}"] = score

            if judge_name == "isomorphic" and accuracy == 1.0:
                scores["slr_bench_solved"] = 1.0

        scores["slr_reward_hacking"] = scores["slr_bench_base"] - scores["slr_bench_isomorphic"]

        selected_key = f"slr_bench_{self.config.slr_reward}"
        selected_score = scores.get(selected_key, 0.0)

        return SLRBenchVerifyResponse(
            **body.model_dump(),
            reward=selected_score,
            slr_bench_isomorphic=scores["slr_bench_isomorphic"],
            slr_bench_base=scores["slr_bench_base"],
            slr_bench_format=scores["slr_bench_format"],
            slr_reward_hacking=scores["slr_reward_hacking"],
            slr_bench_solved=scores["slr_bench_solved"],
            extracted_rule=rule,
        )


if __name__ == "__main__":
    SLRBenchResourcesServer.run_webserver()
