# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent metadata client. Ordinary OpenAI requests perform inference."""

import httpx

from vllm.v1.kv_control.protocol import (
    ActionReceipt,
    CallBinding,
    CapabilitySet,
    ControlTelemetry,
    OperationQuery,
    PolicyRequest,
    PrefixObservation,
    QueryRequest,
    ResolveResult,
    TelemetryRequest,
)


class KVControlClient:
    def __init__(self, client: httpx.AsyncClient):
        """Use a caller-owned HTTP client with base URL and server credentials."""
        self.client = client

    async def capabilities(self) -> CapabilitySet:
        response = await self.client.get("/v1/kv/capabilities")
        response.raise_for_status()
        return CapabilitySet.model_validate(response.json())

    async def query(self, request: QueryRequest) -> PrefixObservation:
        response = await self.client.post(
            "/v1/kv/query", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return PrefixObservation.model_validate(response.json())

    async def apply(self, request: PolicyRequest) -> ActionReceipt:
        # No automatic new action ID on timeout: the caller retries the same
        # immutable command and idempotency key to recover the accepted result.
        response = await self.client.post(
            "/v1/kv/apply", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return ActionReceipt.model_validate(response.json())

    async def status(self, request: OperationQuery) -> ActionReceipt:
        response = await self.client.post(
            "/v1/kv/status", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return ActionReceipt.model_validate(response.json())

    async def resolve(self, request: CallBinding) -> ResolveResult:
        response = await self.client.post(
            "/v1/kv/resolve", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return ResolveResult.model_validate(response.json())

    async def telemetry(self, request: TelemetryRequest) -> ControlTelemetry:
        response = await self.client.post(
            "/v1/kv/telemetry", json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return ControlTelemetry.model_validate(response.json())
