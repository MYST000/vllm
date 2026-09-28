# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boundary validation for the independent engine KV control protocol."""

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm.v1.kv_control.protocol import (
    Action,
    PrefixObservation,
    QueryRequest,
    RestoreCostEstimate,
)

pytestmark = pytest.mark.cpu_test


def test_no_restore_control_action():
    with pytest.raises(ValidationError):
        TypeAdapter(Action).validate_python("RESTORE")


@pytest.mark.asyncio
@pytest.mark.parametrize("api_kind", ["chat", "responses"])
async def test_target_query_rejects_invalid_input_without_echoing_prompt(api_kind):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from fastapi import HTTPException

    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.kv_control.api_router import query_target
    from vllm.v1.kv_control.protocol import TargetQueryRequest

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(engine_client=MagicMock(spec=AsyncLLM))
        )
    )
    payload = TargetQueryRequest(
        owner_scope="s",
        expected_engine_epoch="e",
        query_id="q",
        api_kind=api_kind,
        payload={"model": "m", "messages": "private-input", "input": {"invalid": True}},
    )
    with pytest.raises(HTTPException) as error:
        await query_target(payload, request)
    assert error.value.status_code == 422
    assert error.value.detail == "INVALID_TARGET_REQUEST"


@pytest.mark.asyncio
@pytest.mark.parametrize("api_kind", ["chat", "responses"])
async def test_target_uses_same_adapter_selection_as_inference(api_kind):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.kv_control.api_router import query_target
    from vllm.v1.kv_control.protocol import TargetQueryRequest

    adapter = object()
    serving = SimpleNamespace(
        render_chat_request=AsyncMock(return_value=([], ["rendered"])),
        _make_request=AsyncMock(return_value=([], ["rendered"])),
        _check_model=AsyncMock(return_value=None),
        use_harmony=False,
        default_sampling_params={},
        _maybe_get_adapters=MagicMock(return_value=adapter),
    )
    client = MagicMock(spec=AsyncLLM)
    client.get_supported_tasks = AsyncMock(return_value=("generate",))
    client.input_processor = SimpleNamespace(
        process_inputs=MagicMock(return_value="target")
    )
    client.engine_core = SimpleNamespace(
        call_utility_async=AsyncMock(return_value={"query_id": "q"})
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_client=client,
                openai_serving_chat=serving,
                openai_serving_responses=serving,
            )
        )
    )
    payload = TargetQueryRequest(
        owner_scope="s",
        expected_engine_epoch="e",
        query_id="q",
        api_kind=api_kind,
        payload={
            "model": "m",
            **(
                {"messages": [{"role": "user", "content": "hi"}]}
                if api_kind == "chat"
                else {"input": "hi"}
            ),
        },
    )
    await query_target(payload, request)
    assert serving._maybe_get_adapters.call_args.kwargs == {
        "supports_default_mm_loras": api_kind == "chat"
    }
    assert (
        client.input_processor.process_inputs.call_args.kwargs["lora_request"]
        is adapter
    )


def test_unknown_token_count_cannot_claim_prefill_work():
    observation = dict(
        descriptor_id="d",
        engine_epoch="e",
        state_version=1,
        event_seq=1,
        observed_at_monotonic=1.0,
        prefix_token_count=16,
        gpu_ready_tokens=0,
        recoverable_tokens=None,
        cpu_standalone_tokens=None,
        gpu_resident_blocks_by_group={},
        cpu_ready_blocks_by_group={},
        lookup_state="PENDING",
        lookup_basis="CPUOffloadingSpec",
        reuse_basis="DESCRIPTOR_ONLY",
    )
    result = PrefixObservation.model_validate(observation)
    assert result.prefill_after_restore_tokens is None
    with pytest.raises(ValidationError, match="ID-only"):
        PrefixObservation.model_validate(
            observation | {"prefill_without_restore_tokens": 16}
        )
    with pytest.raises(ValidationError, match="incomplete lookup"):
        PrefixObservation.model_validate(observation | {"recoverable_tokens": 16})


def test_query_rejects_undeclared_count_basis_and_versions():
    query = dict(descriptor_id="d", owner_scope="s", expected_engine_epoch="e")
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(query | {"schema_version": 2})
    with pytest.raises(ValidationError, match="supplied together"):
        QueryRequest.model_validate(query | {"next_prompt_tokens": 32})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(query | {"restore": True})


def test_cost_estimate_requires_source_and_finite_uncertainty():
    cost = dict(
        milliseconds=1.0,
        real_object_bytes=4096,
        basis="CALIBRATED_MODEL",
        source="calibration/1",
        model_version="1",
        observed_at=1.0,
        uncertainty_ms=0.5,
        assumptions="same layout and transfer path",
    )
    assert RestoreCostEstimate.model_validate(cost).milliseconds == 1.0
    with pytest.raises(ValidationError):
        RestoreCostEstimate.model_validate(cost | {"uncertainty_ms": float("inf")})
    del cost["source"]
    with pytest.raises(ValidationError):
        RestoreCostEstimate.model_validate(cost)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, "rank failed"])
async def test_receipt_error_field_is_not_a_transport_error(error):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.kv_control.api_router import _call

    client = MagicMock(spec=AsyncLLM)
    receipt = {"schema_version": 1, "operation_id": "op", "error": error}
    client.engine_core = SimpleNamespace(
        call_utility_async=AsyncMock(return_value=receipt)
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=client))
    )
    assert await _call(request, "apply_kv_policy") == receipt


def test_choice_binding_does_not_mutate_cached_sibling_parameters():
    from vllm import SamplingParams
    from vllm.v1.kv_control.ingress import bind_choice, bind_input

    binding = dict(
        owner_scope="s",
        job_id="j",
        line_id="l",
        request_id="r",
        llm_call_id="c",
        attempt=1,
        context_epoch=0,
    )
    params = SamplingParams(
        n=2, extra_args={"kv_transfer_params": {"kv_control_binding": binding}}
    )
    parent = bind_input(params, 1, 2)
    a = bind_choice(parent, "actual-engine-parent", 0, 2)
    b = bind_choice(parent, "actual-engine-parent", 1, 2)
    left = a.extra_args["kv_transfer_params"]["kv_control_child"]
    right = b.extra_args["kv_transfer_params"]["kv_control_child"]
    assert (left["input_index"], left["output_index"]) == (1, 0)
    assert (right["input_index"], right["output_index"]) == (1, 1)
    assert "kv_control_child" not in parent.extra_args["kv_transfer_params"]
    assert "kv_control_input" not in params.extra_args["kv_transfer_params"]
