# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned metadata contracts, independent of gateway scheduling protocols.

These models do not assert that any engine capability has been implemented.
Ordinary inference is the sole entry point for CPU restoration.
"""

from enum import Enum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

NonNegative = Annotated[int, Field(ge=0, strict=True)]
Positive = Annotated[int, Field(gt=0, strict=True)]
Identifier = Annotated[str, Field(min_length=1)]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1


class KVControlConfig(WireModel):
    enabled: bool = False
    finish_grace_ttl_ms: Annotated[float, Field(ge=0)] = 0
    metadata_ttl_seconds: Annotated[float, Field(gt=0)] = 300
    retention_preferences: bool = True


class OperationQuery(WireModel):
    operation_id: Identifier
    owner_scope: Identifier
    expected_engine_epoch: Identifier


class TelemetryRequest(WireModel):
    owner_scope: Identifier
    expected_engine_epoch: Identifier
    after_event_seq: NonNegative = 0


class ControlEvent(WireModel):
    event_seq: NonNegative
    state_version: NonNegative
    kind: Identifier
    descriptor_id: Identifier | None
    owner_scope: Identifier | None
    engine_epoch: Identifier
    observed_at_monotonic: float
    grace_state: str | None = None
    grace_deadline_monotonic: float | None = None


class ControlTelemetry(WireModel):
    engine_epoch: Identifier
    event_seq: NonNegative
    state_version: NonNegative
    events_gap: bool
    events: tuple[ControlEvent, ...]
    counters: dict[str, float]
    protected_gpu_allocations: NonNegative
    protected_gpu_capacity_bytes: NonNegative
    free_gpu_allocations: NonNegative
    pending_native_jobs: NonNegative
    active_operations: NonNegative
    descriptor_count: NonNegative
    registered_gpu_hashes: NonNegative


class Action(str, Enum):
    KEEP = "KEEP"
    OFFLOAD = "OFFLOAD"
    DROP = "DROP"


class ActionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    APPLIED = "APPLIED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    STALE = "STALE"
    UNSUPPORTED = "UNSUPPORTED"


class GraceState(str, Enum):
    ACTIVE = "ACTIVE"
    HANDED_OFF = "HANDED_OFF"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    INVALIDATED = "INVALIDATED"


class CallBinding(WireModel):
    owner_scope: Identifier
    job_id: Identifier
    line_id: Identifier
    request_id: Identifier
    llm_call_id: Identifier
    attempt: NonNegative
    context_epoch: NonNegative


def validate_transfer_binding(params: dict) -> dict:
    if "kv_control_child" in params or "kv_control_input" in params:
        raise ValueError(
            "KV input/choice identities are reserved for inference ingress"
        )
    if "kv_control_binding" in params:
        CallBinding.model_validate(params["kv_control_binding"])
    return params


KVTransferParams = Annotated[dict, AfterValidator(validate_transfer_binding)]


class EngineIdentity(WireModel):
    instance_id: Identifier
    kv_domain_id: Identifier
    engine_epoch: Identifier
    identity_digest: Identifier
    backend: Identifier
    required_worker_count: Positive


class CapabilitySet(WireModel):
    engine: EngineIdentity
    metadata_ttl_seconds: Annotated[float, Field(gt=0)]
    descriptor_query: bool
    finish_grace: bool
    finish_grace_ttl_ms: Annotated[float, Field(ge=0)]
    gpu_retention_preference: bool
    cpu_backed_eviction_preference: bool
    safe_direct_drop: bool
    cpu_store: bool
    engine_cpu_reuse: bool
    hybrid_state: bool
    transfer_measurement: bool
    restore_cost_estimate: bool
    continuation_proof: bool
    target_prefix_query: bool = True
    offload_gpu_reclaim: bool = True
    layout: dict[str, str | int | list[int]]


class DescriptorHandle(WireModel):
    descriptor_id: Identifier
    engine: EngineIdentity
    binding: CallBinding
    engine_request_id: Identifier
    context_identity_digest: Identifier
    parent_engine_request_id: Identifier | None = None
    output_index: NonNegative = 0
    output_count: Positive = 1
    input_index: NonNegative = 0
    input_count: Positive = 1
    input_token_count: NonNegative
    prefix_token_count: NonNegative
    computed_token_limit: NonNegative
    candidate_token_limit: NonNegative
    state_manifest_ref: Identifier
    expires_at_monotonic: float

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if (
            not self.candidate_token_limit
            <= self.computed_token_limit
            <= (self.prefix_token_count)
        ):
            raise ValueError("candidate <= computed <= described prefix is required")
        return self


class GraceObservation(WireModel):
    grace_id: Identifier
    state: GraceState
    remaining_ms: Annotated[float, Field(ge=0)]
    protected_manifest_ref: Identifier
    protected_bytes_by_group: dict[int, NonNegative]


class RestoreCostEstimate(WireModel):
    milliseconds: Annotated[float, Field(ge=0)]
    real_object_bytes: NonNegative
    basis: Literal["MEASURED_TRANSFER_HISTORY", "CALIBRATED_MODEL"]
    source: Identifier
    model_version: Identifier
    observed_at: float
    uncertainty_ms: Annotated[float, Field(ge=0)]
    assumptions: Identifier


class QueryRequest(WireModel):
    descriptor_id: Identifier
    owner_scope: Identifier
    expected_engine_epoch: Identifier
    skip_reading_prefix_cache: bool = False
    requests_prompt_logprobs: bool = False
    next_prompt_tokens: NonNegative | None = None
    count_basis: Literal["EXACT", "ESTIMATED"] | None = None

    @model_validator(mode="after")
    def require_count_basis(self) -> Self:
        if (self.next_prompt_tokens is None) != (self.count_basis is None):
            raise ValueError(
                "next_prompt_tokens and count_basis must be supplied together"
            )
        return self


class TargetQueryRequest(WireModel):
    owner_scope: Identifier
    expected_engine_epoch: Identifier
    query_id: Identifier
    api_kind: Literal["chat", "responses"]
    payload: dict


class TargetPrefixObservation(WireModel):
    query_id: Identifier
    engine_epoch: Identifier
    engine_identity_digest: Identifier
    state_version: NonNegative
    event_seq: NonNegative
    prompt_tokens: NonNegative
    gpu_ready_tokens: NonNegative
    recoverable_tokens: NonNegative | None
    cpu_load_object_bytes: NonNegative | None
    byte_basis: str = "native objects; common GPU boundary; complete chunks"
    reuse_basis: Literal["TARGET_REQUEST"] = "TARGET_REQUEST"


class PrefixObservation(WireModel):
    descriptor_id: Identifier
    engine_epoch: Identifier
    state_version: NonNegative
    event_seq: NonNegative
    observed_at_monotonic: float
    prefix_token_count: NonNegative
    gpu_ready_tokens: NonNegative
    recoverable_tokens: NonNegative | None
    cpu_standalone_tokens: NonNegative | None
    gpu_resident_blocks_by_group: dict[int, NonNegative]
    cpu_ready_blocks_by_group: dict[int, NonNegative]
    lookup_state: Literal["COMPLETE", "PENDING", "UNSUPPORTED"]
    lookup_basis: Identifier
    reuse_basis: Literal["DESCRIPTOR_ONLY", "ASSUMED_CONTINUATION"]
    next_prompt_tokens: NonNegative | None = None
    count_basis: Literal["EXACT", "ESTIMATED"] | None = None
    prefill_without_restore_tokens: NonNegative | None = None
    prefill_after_restore_tokens: NonNegative | None = None
    grace: GraceObservation | None = None
    restore_cost: RestoreCostEstimate | None = None
    cpu_extension_tokens: NonNegative | None = None
    prefix_uncovered_tokens: NonNegative | None = None
    resume_points_ref: Identifier | None = None
    effective_policy_version: NonNegative | None = None
    effective_policy_action: Action | None = None
    in_flight_operations: tuple[Identifier, ...] = ()
    gpu_retention_bytes: NonNegative = 0
    offload_target_tokens: NonNegative = 0
    offload_object_bytes: NonNegative = 0

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.gpu_ready_tokens > self.prefix_token_count:
            raise ValueError("GPU prefix exceeds descriptor coverage")
        if self.recoverable_tokens is not None and not (
            self.gpu_ready_tokens <= self.recoverable_tokens <= self.prefix_token_count
        ):
            raise ValueError(
                "recoverable range must contain GPU prefix within coverage"
            )
        if self.cpu_standalone_tokens is not None and (
            self.cpu_standalone_tokens > self.prefix_token_count
        ):
            raise ValueError("standalone CPU prefix exceeds descriptor coverage")
        if self.lookup_state != "COMPLETE" and self.recoverable_tokens is not None:
            raise ValueError("incomplete lookup cannot assert a recoverable range")
        if self.next_prompt_tokens is None:
            if any(
                value is not None
                for value in (
                    self.count_basis,
                    self.prefill_without_restore_tokens,
                    self.prefill_after_restore_tokens,
                )
            ):
                raise ValueError("ID-only observations cannot assert next-request work")
        elif self.count_basis is None:
            raise ValueError("next-request counts require a declared basis")
        if self.recoverable_tokens is None and (
            self.prefill_after_restore_tokens is not None
            or self.restore_cost is not None
        ):
            raise ValueError(
                "unknown recovery range cannot carry recovery work or cost"
            )
        return self


class PolicyRequest(WireModel):
    action_id: Identifier
    idempotency_key: Identifier
    action: Action
    descriptor_id: Identifier
    owner_scope: Identifier
    expected_engine_epoch: Identifier
    expected_policy_version: NonNegative
    policy_version: Positive
    source_llm_call_id: Identifier
    expected_tail_request_id: Identifier
    expected_tail_version: NonNegative
    decision_ref: Identifier
    target_resume_token_end: NonNegative | None = None

    @model_validator(mode="after")
    def advancing_policy(self) -> Self:
        if self.policy_version <= self.expected_policy_version:
            raise ValueError("policy version must advance")
        return self


class ActionReceipt(WireModel):
    action_id: Identifier
    idempotency_key: Identifier
    action: Action
    descriptor_id: Identifier
    owner_scope: Identifier
    engine_epoch: Identifier
    status: ActionStatus
    operation_id: Identifier
    applied_policy_version: NonNegative | None
    state_version: NonNegative
    event_seq: NonNegative
    requested_manifest_ref: Identifier
    accepted_manifest_ref: Identifier | None
    completed_manifest_ref: Identifier | None
    target_resume_token_end: NonNegative | None
    gpu_ready_tokens: NonNegative
    recoverable_tokens: NonNegative | None
    cpu_standalone_tokens: NonNegative | None
    cpu_committed_bytes: NonNegative
    cpu_reused_bytes: NonNegative
    gpu_reclaimed_bytes: NonNegative
    skipped_reasons: dict[str, NonNegative]
    error: str | None = None
    grace: GraceObservation | None = None


class ResolveResult(WireModel):
    binding: CallBinding
    status: Literal["PENDING", "READY", "UNKNOWN_BINDING", "DESCRIPTOR_EXPIRED"]
    descriptors: tuple[DescriptorHandle, ...]
    observed_at_monotonic: Annotated[float, Field(ge=0)]
