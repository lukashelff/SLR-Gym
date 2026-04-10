# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

import resources_servers.slr_bench.app as app_module
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.slr_bench.app import (
    SLRBenchResourcesServer,
    SLRBenchResourcesServerConfig,
    SLRBenchVerifyRequest,
    _extract_rule_from_code_blocks,
)


_UNSET = object()


def _build_response(output_text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [
                    {
                        "annotations": [],
                        "text": output_text,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _build_verify_request(
    output_text: str,
    *,
    validation_program: str | None | object = _UNSET,
    evaluation_config: dict[str, str] | None | object = _UNSET,
) -> SLRBenchVerifyRequest:
    if validation_program is _UNSET:
        validation_program = """
            eastbound(train1).
            westbound(train2).
            has_car(train1, car1_1).
            has_car(train2, car2_1).
        """

    if evaluation_config is _UNSET:
        evaluation_config = {
            "positive_predicate": "eastbound",
            "negative_predicate": "westbound",
        }

    return SLRBenchVerifyRequest(
        responses_create_params={
            "input": [
                {
                    "role": "user",
                    "content": "Infer a Prolog rule.",
                }
            ]
        },
        response=_build_response(output_text),
        validation_program=validation_program,
        evaluation_config=evaluation_config,
    )


def _build_server(*, slr_reward: str = "isomorphic") -> SLRBenchResourcesServer:
    return SLRBenchResourcesServer(
        config=SLRBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            slr_reward=slr_reward,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


class TestApp:
    def test_sanity(self) -> None:
        server = _build_server()
        assert isinstance(server, SLRBenchResourcesServer)

    def test_extract_rule_from_code_blocks(self) -> None:
        rule, format_ok = _extract_rule_from_code_blocks("[RULE] eastbound(T) :- has_car(T, C). [/RULE]")
        assert rule == "eastbound(T) :- has_car(T, C)."
        assert format_ok is True

        rule, format_ok = _extract_rule_from_code_blocks("```prolog\neastbound(T) :- has_car(T, C).\n```")
        assert rule == "eastbound(T) :- has_car(T, C)."
        assert format_ok is False

        rule, format_ok = _extract_rule_from_code_blocks("The final rule is eastbound(T) :- has_car(T, C).")
        assert rule is None
        assert format_ok is False

    async def test_verify_dual_scores_and_reward_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_evaluate(*, isomorphic: bool, **kwargs):
            del kwargs
            if isomorphic:
                return {"is_correct": False, "partial_score": 0.6, "syntax_valid": True}
            return {"is_correct": True, "partial_score": 1.0, "syntax_valid": True}

        monkeypatch.setattr(app_module, "evaluate_prediction", fake_evaluate)

        body = _build_verify_request("[RULE] eastbound(T) :- has_car(T, C). [/RULE]")

        server_iso = _build_server(slr_reward="isomorphic")
        result_iso = await server_iso.verify(body)
        assert result_iso.reward == pytest.approx(result_iso.slr_bench_isomorphic)

        server_base = _build_server(slr_reward="base")
        result_base = await server_base.verify(body)
        assert result_base.reward == pytest.approx(result_base.slr_bench_base)

        assert result_base.slr_bench_base > result_base.slr_bench_isomorphic
        assert result_base.slr_reward_hacking == pytest.approx(
            result_base.slr_bench_base - result_base.slr_bench_isomorphic
        )
        assert result_base.slr_bench_solved == 0.0

    async def test_verify_code_block_and_format_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            app_module,
            "evaluate_prediction",
            lambda **kwargs: {"is_correct": True, "partial_score": 1.0, "syntax_valid": True},
        )

        body_rule_tag = _build_verify_request("[RULE] eastbound(T) :- has_car(T, C). [/RULE]")
        body_fenced = _build_verify_request("```prolog\neastbound(T) :- has_car(T, C).\n```")

        server = _build_server()
        result_rule_tag = await server.verify(body_rule_tag)
        result_fenced = await server.verify(body_fenced)

        assert result_rule_tag.reward > 0.0
        assert result_fenced.reward > 0.0
        assert result_rule_tag.slr_bench_format == 1.0
        assert result_fenced.slr_bench_format == 0.0

    async def test_failure_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _build_server()

        empty_body = _build_verify_request("")
        empty_result = await server.verify(empty_body)
        assert empty_result.reward == 0.0

        missing_program_body = _build_verify_request(
            "[RULE] eastbound(T) :- has_car(T, C). [/RULE]",
            validation_program=None,
        )
        missing_program_result = await server.verify(missing_program_body)
        assert missing_program_result.reward == 0.0

        missing_eval_config_body = _build_verify_request(
            "[RULE] eastbound(T) :- has_car(T, C). [/RULE]",
            evaluation_config=None,
        )
        missing_eval_config_result = await server.verify(missing_eval_config_body)
        assert missing_eval_config_result.reward == 0.0

        prose_body = _build_verify_request("eastbound(T) :- has_car(T, C).")
        prose_result = await server.verify(prose_body)
        assert prose_result.reward == 0.0

        def raise_timeout(**kwargs):
            del kwargs
            raise TimeoutError("timed out")

        monkeypatch.setattr(app_module, "evaluate_prediction", raise_timeout)
        timeout_body = _build_verify_request("[RULE] eastbound(T) :- has_car(T, C). [/RULE]")
        timeout_result = await server.verify(timeout_body)
        assert timeout_result.reward == 0.0
        assert timeout_result.slr_bench_isomorphic == 0.0
        assert timeout_result.slr_bench_base == 0.0
