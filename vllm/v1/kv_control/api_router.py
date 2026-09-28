# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finite metadata endpoints using the existing EngineCore utility transport."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_control.protocol import (
    CallBinding,
    OperationQuery,
    PolicyRequest,
    QueryRequest,
    TargetQueryRequest,
    TelemetryRequest,
    WireModel,
)

router = APIRouter(prefix="/v1/kv", tags=["KV control v1"])


async def _call(request: Request, method: str, payload: WireModel | None = None):
    args = () if payload is None else (payload.model_dump(mode="json"),)
    return await _utility(request, method, *args)


async def _utility(request: Request, method: str, *args):
    client = request.app.state.engine_client
    if not isinstance(client, AsyncLLM):
        raise HTTPException(501, "UNSUPPORTED_ENGINE_CLIENT")
    result = await client.engine_core.call_utility_async(method, *args)
    if "kv_control_error" in result:
        code = result["kv_control_error"]
        status = (
            501
            if code == "KV_CONTROL_DISABLED"
            else 403
            if code == "OWNER_MISMATCH"
            else 410
            if code.endswith("EXPIRED")
            else 409
        )
        raise HTTPException(status, code)
    return result


@router.post("/query-target")
async def query_target(payload: TargetQueryRequest, request: Request) -> dict[str, Any]:
    """Render the actual input; observe caches without submitting or acquiring it."""
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.entrypoints.serve.engine.protocol import ErrorResponse

    client = request.app.state.engine_client
    if not isinstance(client, AsyncLLM):
        raise HTTPException(501, "UNSUPPORTED_ENGINE_CLIENT")
    try:
        parsed = (
            ChatCompletionRequest.model_validate(payload.payload)
            if payload.api_kind == "chat"
            else ResponsesRequest.model_validate(payload.payload)
        )
    except ValidationError as exc:
        # Do not include input values (potentially full prompts) in diagnostics.
        raise HTTPException(422, "INVALID_TARGET_REQUEST") from exc
    if isinstance(parsed, ChatCompletionRequest):
        serving = request.app.state.openai_serving_chat
        rendered = await serving.render_chat_request(parsed)
    else:
        serving = request.app.state.openai_serving_responses
        error = await serving._check_model(parsed)
        if error is not None:
            raise HTTPException(error.error.code, error.error.message)
        previous = None
        if parsed.previous_response_id is not None:
            async with serving.response_store_lock:
                previous = serving.response_store.get(parsed.previous_response_id)
            if previous is None:
                raise HTTPException(404, "PREVIOUS_RESPONSE_NOT_FOUND")
        rendered = (
            serving._make_request_with_harmony(parsed, previous)
            if serving.use_harmony
            else await serving._make_request(parsed, previous)
        )
    if isinstance(rendered, ErrorResponse):
        raise HTTPException(rendered.error.code, rendered.error.message)
    _, inputs = rendered
    observations = []
    for index, engine_input in enumerate(inputs):
        # Output generation is not run. The same input processor applies native
        # truncation, LoRA, multimodal hashes and cache salt before the lookup.
        params = parsed.to_sampling_params(1, serving.default_sampling_params)
        target = client.input_processor.process_inputs(
            f"{payload.query_id}:{index}",
            engine_input,
            params,
            supported_tasks=await client.get_supported_tasks(),
            lora_request=serving._maybe_get_adapters(
                parsed,
                supports_default_mm_loras=isinstance(parsed, ChatCompletionRequest),
            ),
        )
        observations.append(
            await _utility(
                request,
                "query_target_prefix",
                target,
                payload.expected_engine_epoch,
                payload.query_id,
            )
        )
    return {
        "schema_version": 1,
        "query_id": payload.query_id,
        "engine_epoch": payload.expected_engine_epoch,
        "inputs": observations,
    }


@router.get("/capabilities")
async def capabilities(request: Request) -> dict[str, Any]:
    return await _call(request, "kv_control_capabilities")


@router.post("/query")
async def query(payload: QueryRequest, request: Request) -> dict[str, Any]:
    return await _call(request, "query_kv_descriptor", payload)


@router.post("/apply")
async def apply(payload: PolicyRequest, request: Request) -> dict[str, Any]:
    return await _call(request, "apply_kv_policy", payload)


@router.post("/status")
async def status(payload: OperationQuery, request: Request) -> dict[str, Any]:
    return await _call(request, "kv_operation_status", payload)


@router.post("/resolve")
async def resolve(payload: CallBinding, request: Request) -> dict[str, Any]:
    return await _call(request, "resolve_finished_descriptor", payload)


@router.post("/telemetry")
async def telemetry(payload: TelemetryRequest, request: Request) -> dict[str, Any]:
    return await _call(request, "kv_control_telemetry", payload)
