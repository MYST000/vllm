# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-thread descriptor registry, finish protection, and retention control.

All physical storage, prefix lookup and DMA remain in the native cache managers.
This module owns metadata and explicit references only; it never restores KV.
"""

import copy
import hashlib
import heapq
import json
import math
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    DescriptorStoreSource,
    HashPrefix,
    OffloadingConnectorScheduler,
    offloadable_token_limit,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_control.protocol import (
    Action,
    ActionReceipt,
    ActionStatus,
    CallBinding,
    CapabilitySet,
    ControlEvent,
    ControlTelemetry,
    DescriptorHandle,
    EngineIdentity,
    GraceObservation,
    GraceState,
    KVControlConfig,
    OperationQuery,
    PolicyRequest,
    PrefixObservation,
    QueryRequest,
    ResolveResult,
    TargetPrefixObservation,
    TelemetryRequest,
)
from vllm.v1.kv_offload.base import OffloadKey, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.request import Request, RequestStatus

BINDING_KEY = "kv_control_binding"
METADATA_KEY = "kv_control_v1"
Allocation = tuple[int, int]


class KVControlError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ObjectRef:
    key: OffloadKey
    group: int
    token_end: int
    gpu_hashes: tuple[BlockHashWithGroupId, ...]
    first_block_index: int
    num_bytes: int


@dataclass
class FinishGrace:
    grace_id: str
    deadline: float
    held: dict[Allocation, tuple[KVCacheBlock, int]]
    state: GraceState = GraceState.ACTIVE
    required_hashes: dict[Allocation, set[BlockHashWithGroupId]] = field(
        default_factory=dict
    )
    handed_off: set[BlockHashWithGroupId] = field(default_factory=set)


@dataclass
class Descriptor:
    handle: DescriptorHandle
    hashes: list[BlockHash]
    context: ReqContext
    objects: dict[OffloadKey, ObjectRef]
    grace: FinishGrace | None
    policy: PolicyRequest | None = None
    policy_version: int = 0
    invalidated: bool = False
    observed_allocations: set[Allocation] = field(default_factory=set)


@dataclass
class Operation:
    command: PolicyRequest
    receipt: ActionReceipt
    target_keys: frozenset[OffloadKey]
    newly_stored: dict[OffloadKey, int] = field(default_factory=dict)
    reused_bytes: int = 0
    committed_bytes: int = 0
    errors: list[str] = field(default_factory=list)
    job_bytes: dict[int, int] = field(default_factory=dict)
    pending_reuse_bytes: dict[int, int] = field(default_factory=dict)
    gpu_targets: dict[Allocation, int] = field(default_factory=dict)
    gpu_reclaimed_bytes: int = 0


class ReadyBlockPool(BlockPool):
    """Read-only view used by the same native coordinator lookup algorithms."""

    def __init__(self, manager: "KVControlManager"):
        self.manager = manager
        self.hash_block_size = manager.pool.hash_block_size
        self.null_block = manager.pool.null_block

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        result = []
        for group in kv_cache_group_ids:
            replicas = self.manager.gpu_replicas(
                make_block_hash_with_group_id(block_hash, group)
            )
            if not replicas:
                return None
            result.append(replicas[0])
        return result


class KVControlManager:
    def __init__(
        self,
        config: KVControlConfig,
        vllm_config: VllmConfig,
        cache: KVCacheManager,
        connector: OffloadingConnectorScheduler,
    ):
        if not isinstance(connector.manager, CPUOffloadingManager):
            raise KVControlError("UNSUPPORTED_BACKEND")
        parallel = vllm_config.parallel_config
        groups = cache.kv_cache_config.kv_cache_groups
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or vllm_config.speculative_config is not None
            or connector.canonical_layout
            or not cache.enable_caching
            or any(
                not isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec))
                for g in groups
            )
            or any(
                isinstance(g.kv_cache_spec, MambaSpec)
                and g.kv_cache_spec.mamba_cache_mode != "align"
                for g in groups
            )
        ):
            raise KVControlError("UNSUPPORTED_CONTROL_LAYOUT")
        self.config = config
        self.cache = cache
        self.pool = cache.block_pool
        self.connector = connector
        self.cpu = connector.manager
        self.descriptors: dict[str, Descriptor] = {}
        self.bindings: dict[str, dict[str, str | None]] = {}
        self.binding_expected: dict[str, int] = {}
        self.request_bindings: dict[str, CallBinding] = {}
        self.line_calls: dict[tuple[str, str, str], str] = {}
        self.active_lines: Counter[tuple[str, str, str]] = Counter()
        self.by_line: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self.operations: dict[str, Operation] = {}
        self.active_operations: set[str] = set()
        self.idempotency: dict[tuple[str, str], str] = {}
        self.action_ids: dict[tuple[str, str], str] = {}
        self.deadlines: list[tuple[float, str, str]] = []
        self.events: deque[dict[str, Any]] = deque()
        self.state_version = 0
        self.event_seq = 0
        self.residency_changed = False
        self.step_seq = 0
        self.scheduling_seq: int | None = None
        self.pending_writes: dict[Allocation, int] = {}
        self.writes_by_step: dict[int, set[Allocation]] = defaultdict(set)
        self.dirty_descriptors: set[str] = set()
        self.by_gpu_hash: dict[BlockHashWithGroupId, set[str]] = defaultdict(set)
        self.by_cpu_key: dict[OffloadKey, set[str]] = defaultdict(set)
        self.drop_gpu: dict[str, dict[Allocation, int]] = {}
        self.drop_cpu: dict[str, dict[OffloadKey, int]] = {}
        self.offload_gpu: dict[str, str] = {}
        self.cpu_backed: dict[str, bool] = {}
        self.by_allocation: dict[Allocation, set[str]] = defaultdict(set)
        self.free_heap: list[tuple[int, int, int, int, int]] = []
        self.free_versions: dict[int, int] = {}
        self.free_recency: dict[int, int] = {}
        self.free_age = 0
        self.grace_references: Counter[Allocation] = Counter()
        self.metrics: dict[str, float] = defaultdict(float)
        self.gpu_allocation_bytes = connector.gpu_bytes_per_allocation
        self.group_bytes = {
            i: (
                group.kv_cache_spec.state_content_size_bytes
                if isinstance(group.kv_cache_spec, MambaSpec)
                else cast(
                    FullAttentionSpec, group.kv_cache_spec
                ).unpadded_page_size_bytes
            )
            * len(group.layer_names)
            * parallel.world_size
            for i, group in enumerate(groups)
        }
        self.cpu_group_bytes = {
            group: size // parallel.world_size if connector.replicated_layout else size
            for group, size in self.group_bytes.items()
        }
        identity = {
            "model": vllm_config.model_config.model,
            "revision": vllm_config.model_config.revision,
            "dtype": str(vllm_config.model_config.dtype),
            "config_hash": vllm_config.compute_hash(),
            "hash_tokens": cache.block_pool.hash_block_size,
            "groups": [repr(group.kv_cache_spec) for group in groups],
            "tp": parallel.tensor_parallel_size,
        }
        assert vllm_config.kv_transfer_config is not None
        engine_id = vllm_config.kv_transfer_config.engine_id
        assert engine_id is not None
        self.identity = EngineIdentity(
            instance_id=engine_id,
            kv_domain_id=engine_id,
            engine_epoch=uuid.uuid4().hex,
            identity_digest=hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest(),
            backend="CPUOffloadingSpec",
            required_worker_count=connector.config.num_workers,
        )
        self.pool.state_observer = self._gpu_changed
        self.cpu.state_observer = self._cpu_changed
        if config.retention_preferences:
            for block in self.pool.blocks:
                if not block.ref_cnt and not block.is_null:
                    self._index_free(block)
            self.pool.free_block_selector = self._select_free

    @staticmethod
    def _binding_key(binding: CallBinding) -> str:
        return binding.model_dump_json()

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            engine=self.identity,
            metadata_ttl_seconds=self.config.metadata_ttl_seconds,
            descriptor_query=True,
            finish_grace=True,
            finish_grace_ttl_ms=self.config.finish_grace_ttl_ms,
            gpu_retention_preference=self.config.retention_preferences,
            cpu_backed_eviction_preference=self.config.retention_preferences,
            safe_direct_drop=True,
            cpu_store=self.config.retention_preferences,
            engine_cpu_reuse=True,
            hybrid_state=True,
            transfer_measurement=True,
            restore_cost_estimate=False,
            continuation_proof=False,
            layout={
                "tp": self.identity.required_worker_count,
                "pp": 1,
                "dp": 1,
                "group_block_tokens": [
                    g.tokens_per_block for g in self.connector.config.kv_group_configs
                ],
                "binding": BINDING_KEY,
            },
        )

    def _event(self, kind: str, descriptor_id: str | None = None) -> None:
        self.state_version += 1
        self.event_seq += 1
        descriptor = self.descriptors.get(descriptor_id) if descriptor_id else None
        self.events.append(
            dict(
                event_seq=self.event_seq,
                state_version=self.state_version,
                kind=kind,
                descriptor_id=descriptor_id,
                engine_epoch=self.identity.engine_epoch,
                owner_scope=descriptor.handle.binding.owner_scope
                if descriptor
                else None,
                grace_state=descriptor.grace.state
                if descriptor and descriptor.grace
                else None,
                grace_deadline_monotonic=descriptor.grace.deadline
                if descriptor and descriptor.grace
                else None,
                observed_at_monotonic=time.monotonic(),
            )
        )

    def _gpu_changed(self, block: KVCacheBlock, change: str) -> None:
        allocation = (block.block_id, block.allocation_generation)
        self.dirty_descriptors.update(self.by_allocation.get(allocation, ()))
        if change == "HASH_ADDED" and self.scheduling_seq is not None:
            self.pending_writes[allocation] = self.scheduling_seq
            self.writes_by_step[self.scheduling_seq].add(allocation)
        hashes = self.pool.cached_block_hashes_by_block.get(
            block.block_id, set()
        ).copy()
        if block.block_hash is not None:
            hashes.add(block.block_hash)
        for key in hashes:
            self.dirty_descriptors.update(self.by_gpu_hash.get(key, ()))
        # Removed hashes cannot be read from the block now. The allocation
        # reverse index for pending DROP/GRACE is processed at engine safe points.
        self.state_version += 1
        self.residency_changed = True
        if (
            not block.is_null
            and block.ref_cnt == 0
            and change not in ("HASH_REMOVED", "HASH_REMOVING")
        ):
            self._index_free(block)
        elif change == "ACQUIRED":
            self.free_versions.pop(block.block_id, None)
            self.free_recency.pop(block.block_id, None)

    def _cpu_changed(self, keys: Collection[OffloadKey]) -> None:
        self.state_version += 1
        self.residency_changed = True
        for key in keys:
            self.dirty_descriptors.update(self.by_cpu_key.get(key, ()))

    def begin_step(self) -> None:
        self.poll()
        self.step_seq += 1
        self.scheduling_seq = self.step_seq

    def end_schedule(self, output: Any) -> None:
        output.kv_control_step = self.scheduling_seq
        for block_copy in output.kv_cache_block_copies or ():
            block = self.pool.blocks[block_copy.dst_block_id]
            allocation = (block.block_id, block.allocation_generation)
            self.pending_writes[allocation] = self.step_seq
            self.writes_by_step[self.step_seq].add(allocation)
        self.scheduling_seq = None

    def complete_step(self, output: Any) -> None:
        seq = output.kv_control_step
        if seq is not None:
            for allocation in self.writes_by_step.pop(seq, ()):
                if self.pending_writes.get(allocation) == seq:
                    del self.pending_writes[allocation]
                    block = self.pool.blocks[allocation[0]]
                    if block.allocation_generation == allocation[1]:
                        self._gpu_changed(block, "WRITE_COMPLETED")

    def gpu_replicas(self, key: BlockHashWithGroupId) -> tuple[KVCacheBlock, ...]:
        return tuple(
            block
            for block in self.pool.cached_block_hash_to_block.get_all_blocks(key)
            if not block.is_null
            and (block.block_id, block.allocation_generation) not in self.pending_writes
        )

    def on_request(self, request: Request) -> None:
        params = request.kv_transfer_params or {}
        raw = params.get(BINDING_KEY)
        if raw is None:
            return
        binding = CallBinding.model_validate(raw)
        self.request_bindings[request.request_id] = binding
        self.bindings.setdefault(self._binding_key(binding), {})[request.request_id] = (
            None
        )
        child = params.get("kv_control_child", {})
        self.binding_expected[self._binding_key(binding)] = child.get(
            "input_count", 1
        ) * child.get("output_count", 1)
        line = (binding.owner_scope, binding.job_id, binding.line_id)
        self.active_lines[line] += 1
        previous = self.line_calls.get(line)
        if previous is not None and previous != binding.llm_call_id:
            for did in self.by_line.get(line, ()):
                descriptor = self.descriptors[did]
                descriptor.invalidated = True
                self._end_grace(descriptor, GraceState.CANCELLED)
                self._policy_changed(descriptor)
                self._event("POLICY_INVALIDATED", did)
        self.line_calls[line] = binding.llm_call_id

    def _objects(
        self, hashes: list[BlockHash], limit: int
    ) -> dict[OffloadKey, ObjectRef]:
        objects = {}
        hash_tokens = self.connector.config.tokens_per_hash
        for group in self.connector.config.kv_group_configs:
            step = (
                hash_tokens
                if self.connector.config.supports_partial_tail
                else group.tokens_per_block
            )
            for end in range(step, limit + 1, step):
                if end // hash_tokens > len(hashes):
                    break
                block_size = group.tokens_per_block
                if (
                    end % group.tokens_per_chunk
                    and not self.connector.config.supports_partial_tail
                ):
                    # GPU retention also covers blocks that cannot form a CPU chunk.
                    begin = end - block_size
                    ends = [end]
                elif end % group.tokens_per_chunk:
                    # Native partial-tail storage consists of the final physical
                    # attention block and the exact recurrent checkpoint.
                    begin = end // block_size * block_size
                    ends = [end]
                elif group.requires_cow_source:
                    begin = end - block_size
                    ends = [end]
                else:
                    begin = end - group.tokens_per_chunk
                    ends = list(range(begin + block_size, end + 1, block_size))
                key = make_offload_key(hashes[end // hash_tokens - 1], group.group_idx)
                objects[key] = ObjectRef(
                    key,
                    group.group_idx,
                    end,
                    tuple(
                        make_block_hash_with_group_id(
                            hashes[n // hash_tokens - 1], group.group_idx
                        )
                        for n in ends
                    ),
                    begin // block_size,
                    len(ends) * self.cpu_group_bytes[group.group_idx],
                )
        return objects

    def finish(self, request: Request) -> dict[str, Any] | None:
        binding = self.request_bindings.pop(request.request_id, None)
        if binding is None:
            return None
        line = (binding.owner_scope, binding.job_id, binding.line_id)
        self.active_lines[line] -= 1
        if not self.active_lines[line]:
            del self.active_lines[line]
        started = time.monotonic()
        computed = max(0, request.num_computed_tokens - request.num_in_flight_tokens)
        computed = min(computed, request.num_tokens)
        candidate = offloadable_token_limit(
            computed,
            request.num_tokens,
            request.num_prompt_tokens,
            self.connector.config.offload_prompt_only,
            (request.kv_transfer_params or {}).get("max_offload_tokens"),
        )
        descriptor_id = uuid.uuid4().hex
        handle = DescriptorHandle(
            descriptor_id=descriptor_id,
            engine=self.identity,
            binding=binding,
            engine_request_id=request.request_id,
            context_identity_digest=hashlib.sha256(
                self.identity.identity_digest.encode() + b"".join(request.block_hashes)
            ).hexdigest(),
            **(request.kv_transfer_params or {}).get("kv_control_child", {}),
            input_token_count=request.num_prompt_tokens,
            prefix_token_count=request.num_tokens,
            computed_token_limit=computed,
            candidate_token_limit=candidate,
            state_manifest_ref=descriptor_id + ":manifest",
            expires_at_monotonic=started + self.config.metadata_ttl_seconds,
        )
        hashes = list(request.block_hashes)
        descriptor = Descriptor(
            handle,
            hashes,
            ReqContext(descriptor_id),
            self._objects(hashes, computed),
            None,
            invalidated=self.line_calls.get(line) != binding.llm_call_id,
        )
        self.descriptors[descriptor_id] = descriptor
        self.by_line[(binding.owner_scope, binding.job_id, binding.line_id)].add(
            descriptor_id
        )
        self.bindings[self._binding_key(binding)][request.request_id] = descriptor_id
        for obj in descriptor.objects.values():
            self.by_cpu_key[obj.key].add(descriptor_id)
            for key in obj.gpu_hashes:
                self.by_gpu_hash[key].add(descriptor_id)
        successful = request.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_REPETITION,
        )
        self.metrics["descriptor_registration_seconds"] += time.monotonic() - started
        grace_started = time.monotonic()
        if (
            successful
            and not descriptor.invalidated
            and self.config.finish_grace_ttl_ms > 0
        ):
            held = {}
            required: dict[Allocation, set[BlockHashWithGroupId]] = defaultdict(set)
            for obj in descriptor.objects.values():
                for key in obj.gpu_hashes:
                    for block in self.gpu_replicas(key):
                        allocation = (block.block_id, block.allocation_generation)
                        held[allocation] = (block, obj.group)
                        required[allocation].add(key)
                        self.by_allocation[allocation].add(descriptor_id)
                        descriptor.observed_allocations.add(allocation)
            self.pool.touch([block for block, _ in held.values()])
            self.grace_references.update(held.keys())
            self.metrics["grace_capacity_bytes_high_watermark"] = max(
                self.metrics["grace_capacity_bytes_high_watermark"],
                len(self.grace_references) * self.gpu_allocation_bytes,
            )
            deadline = started + self.config.finish_grace_ttl_ms / 1000
            descriptor.grace = FinishGrace(
                uuid.uuid4().hex, deadline, held, required_hashes=dict(required)
            )
            heapq.heappush(self.deadlines, (deadline, "GRACE", descriptor_id))
        self.metrics["grace_acquire_seconds"] += time.monotonic() - grace_started
        heapq.heappush(
            self.deadlines, (handle.expires_at_monotonic, "METADATA", descriptor_id)
        )
        self._event("DESCRIPTOR_CREATED", descriptor_id)
        self.metrics["finish_count"] += 1
        self.metrics["finish_seconds"] += time.monotonic() - started
        observation = self.grace_observation(descriptor)
        return {
            "descriptor": handle.model_dump(mode="json"),
            "grace": observation.model_dump(mode="json") if observation else None,
        }

    def grace_observation(self, descriptor: Descriptor) -> GraceObservation | None:
        grace = descriptor.grace
        if grace is None:
            return None
        sizes: Counter[int] = Counter()
        for _, group in grace.held.values():
            sizes[group] += self.group_bytes[group]
        return GraceObservation(
            grace_id=grace.grace_id,
            state=grace.state,
            remaining_ms=max(0, (grace.deadline - time.monotonic()) * 1000)
            if grace.state is GraceState.ACTIVE
            else 0,
            protected_manifest_ref=descriptor.handle.state_manifest_ref + ":grace",
            protected_bytes_by_group=dict(sizes),
        )

    def _end_grace(
        self,
        descriptor: Descriptor,
        state: GraceState,
        allocations: set[Allocation] | None = None,
    ) -> None:
        grace = descriptor.grace
        if grace is None or grace.state is not GraceState.ACTIVE:
            return
        started = time.monotonic()
        selected = (
            set(grace.held) if allocations is None else allocations & grace.held.keys()
        )
        blocks = []
        for allocation in selected:
            block, _ = grace.held.pop(allocation)
            self.grace_references[allocation] -= 1
            if self.grace_references[allocation] == 0:
                del self.grace_references[allocation]
            assert block.allocation_generation == allocation[1]
            blocks.append(block)
        self.pool.free_blocks(blocks)
        self.metrics["grace_release_seconds"] += time.monotonic() - started
        self.metrics["grace_released_references"] += len(blocks)
        if not grace.held:
            grace.state = state
            self._event("GRACE_ENDED", descriptor.handle.descriptor_id)

    def next_timeout(self) -> float | None:
        return (
            max(0, self.deadlines[0][0] - time.monotonic()) if self.deadlines else None
        )

    def _get(self, descriptor_id: str, owner: str, epoch: str) -> Descriptor:
        if epoch != self.identity.engine_epoch:
            raise KVControlError("STALE_ENGINE_EPOCH")
        descriptor = self.descriptors.get(descriptor_id)
        if descriptor is None:
            raise KVControlError("DESCRIPTOR_EXPIRED")
        if descriptor.handle.binding.owner_scope != owner:
            raise KVControlError("OWNER_MISMATCH")
        if descriptor.handle.engine.engine_epoch != epoch:
            raise KVControlError("STALE_ENGINE_EPOCH")
        if time.monotonic() >= descriptor.handle.expires_at_monotonic:
            raise KVControlError("DESCRIPTOR_EXPIRED")
        return descriptor

    def resolve(self, binding: CallBinding) -> ResolveResult:
        self.poll()
        observed_at = time.monotonic()
        entries = self.bindings.get(self._binding_key(binding))
        if entries is None:
            return ResolveResult(
                binding=binding,
                status="UNKNOWN_BINDING",
                descriptors=(),
                observed_at_monotonic=observed_at,
            )
        handles = tuple(
            self.descriptors[did].handle
            for did in entries.values()
            if did in self.descriptors
            and self.descriptors[did].handle.expires_at_monotonic > observed_at
            and self.descriptors[did].handle.engine.engine_epoch
            == self.identity.engine_epoch
        )
        status = (
            "PENDING"
            if (
                any(did is None for did in entries.values())
                or len(entries) < self.binding_expected[self._binding_key(binding)]
            )
            else "READY"
            if handles
            else "DESCRIPTOR_EXPIRED"
        )
        return ResolveResult(
            binding=binding,
            status=status,
            descriptors=handles,
            observed_at_monotonic=observed_at,
        )

    def query(self, query: QueryRequest) -> PrefixObservation:
        self.poll()
        descriptor = self._get(
            query.descriptor_id, query.owner_scope, query.expected_engine_epoch
        )
        return self._observe(
            descriptor,
            query.next_prompt_tokens,
            query.count_basis,
            skip_cache=query.skip_reading_prefix_cache
            or query.requests_prompt_logprobs,
        )

    def telemetry(self, request: TelemetryRequest) -> ControlTelemetry:
        self.poll()
        if request.expected_engine_epoch != self.identity.engine_epoch:
            raise KVControlError("STALE_ENGINE_EPOCH")
        return ControlTelemetry(
            engine_epoch=self.identity.engine_epoch,
            event_seq=self.event_seq,
            state_version=self.state_version,
            events_gap=request.after_event_seq
            < (self.events[0]["event_seq"] - 1 if self.events else self.event_seq),
            events=tuple(
                ControlEvent.model_validate(event)
                for event in self.events
                if event["event_seq"] > request.after_event_seq
                and event["owner_scope"] in (None, request.owner_scope)
            ),
            counters=dict(self.metrics),
            protected_gpu_allocations=len(self.grace_references),
            protected_gpu_capacity_bytes=len(self.grace_references)
            * self.gpu_allocation_bytes,
            free_gpu_allocations=self.pool.get_num_free_blocks(),
            pending_native_jobs=len(self.connector._jobs),
            active_operations=len(self.active_operations),
            descriptor_count=len(self.descriptors),
            registered_gpu_hashes=len(self.by_gpu_hash),
        )

    def query_target(
        self, request: Request, expected_engine_epoch: str, query_id: str
    ) -> TargetPrefixObservation:
        self.poll()
        if expected_engine_epoch != self.identity.engine_epoch:
            raise KVControlError("STALE_ENGINE_EPOCH")
        p = request.num_prompt_tokens
        coordinator = copy.copy(self.cache.coordinator)
        coordinator.block_pool = ReadyBlockPool(self)
        _, gpu, _ = coordinator.find_longest_cache_hit(
            request.block_hashes, max(0, p - 1)
        )
        context = ReqContext(request.request_id)
        prefix = HashPrefix(request.request_id, request.block_hashes, p, p)
        extension = self.connector.query_prefix(prefix, gpu, context)
        total = None if extension is None else gpu + extension
        params = request.sampling_params
        if request.skip_reading_prefix_cache or (
            params is not None and params.prompt_logprobs is not None
        ):
            gpu = total = 0
        load_bytes = None
        if total is not None:
            objects = self._objects(request.block_hashes, total)
            keys = self._transfer_keys(request.block_hashes, gpu, total)
            if all(
                key in objects
                and (state := self.cpu.replica_state(key)) is not None
                and state[2]
                for key in keys
            ):
                load_bytes = sum(objects[key].num_bytes for key in keys)
        return TargetPrefixObservation(
            query_id=query_id,
            engine_epoch=self.identity.engine_epoch,
            engine_identity_digest=self.identity.identity_digest,
            state_version=self.state_version,
            event_seq=self.event_seq,
            prompt_tokens=p,
            gpu_ready_tokens=gpu,
            recoverable_tokens=total,
            cpu_load_object_bytes=load_bytes,
        )

    def _transfer_keys(
        self, hashes: list[BlockHash], local: int, target: int
    ) -> set[OffloadKey]:
        keys: set[OffloadKey] = set()
        if target <= local:
            return keys
        hash_tokens = self.connector.config.tokens_per_hash
        for group in self.connector.config.kv_group_configs:
            if group.requires_cow_source:
                ends = [target]
            else:
                chunk = group.tokens_per_chunk
                ends = list(range((local // chunk + 1) * chunk, target + 1, chunk))
                if target % chunk:
                    ends.append(target)
            keys.update(
                make_offload_key(hashes[end // hash_tokens - 1], group.group_idx)
                for end in ends
                if end
            )
        return keys

    def _observe(
        self,
        descriptor: Descriptor,
        next_tokens: int | None = None,
        count_basis: Literal["EXACT", "ESTIMATED"] | None = None,
        skip_cache: bool = False,
    ) -> PrefixObservation:
        started = time.monotonic()
        limit = descriptor.handle.computed_token_limit
        if next_tokens is not None:
            limit = min(limit, max(0, next_tokens - 1))
        coordinator = copy.copy(self.cache.coordinator)
        coordinator.block_pool = ReadyBlockPool(self)
        _, gpu_hit, _ = coordinator.find_longest_cache_hit(descriptor.hashes, limit)
        prefix = HashPrefix(
            descriptor.handle.descriptor_id, descriptor.hashes, limit + 1, limit + 1
        )
        # OffloadingConnector requires a common local boundary across all groups.
        extension = self.connector.query_prefix(prefix, gpu_hit, descriptor.context)
        cpu_hit = self.connector.query_prefix(prefix, 0, descriptor.context)
        recoverable = None if extension is None else min(limit, gpu_hit + extension)
        if skip_cache:
            gpu_hit = recoverable = cpu_hit = 0
        gpu_counts: dict[int, set[Allocation]] = defaultdict(set)
        cpu_counts: Counter[int] = Counter()
        for obj in descriptor.objects.values():
            for key in obj.gpu_hashes:
                gpu_counts[obj.group].update(
                    (b.block_id, b.allocation_generation)
                    for b in self.gpu_replicas(key)
                )
            state = self.cpu.replica_state(obj.key)
            if state is not None and state[2]:
                cpu_counts[obj.group] += 1
        self.metrics["query_count"] += 1
        self.metrics["query_seconds"] += time.monotonic() - started
        alignment = (
            self.connector.config.tokens_per_hash
            if self.connector.config.supports_partial_tail
            else math.lcm(
                *(g.tokens_per_chunk for g in self.connector.config.kv_group_configs)
            )
        )
        offload_target = (
            descriptor.handle.candidate_token_limit // alignment * alignment
        )
        offload_keys = self._transfer_keys(descriptor.hashes, 0, offload_target)
        return PrefixObservation(
            descriptor_id=descriptor.handle.descriptor_id,
            engine_epoch=self.identity.engine_epoch,
            state_version=self.state_version,
            event_seq=self.event_seq,
            observed_at_monotonic=time.monotonic(),
            prefix_token_count=descriptor.handle.prefix_token_count,
            cpu_extension_tokens=recoverable - gpu_hit
            if recoverable is not None
            else None,
            prefix_uncovered_tokens=descriptor.handle.prefix_token_count - recoverable
            if recoverable is not None
            else None,
            resume_points_ref=descriptor.handle.state_manifest_ref,
            effective_policy_version=descriptor.policy_version
            if descriptor.policy is not None and not descriptor.invalidated
            else None,
            effective_policy_action=descriptor.policy.action
            if descriptor.policy is not None and not descriptor.invalidated
            else None,
            in_flight_operations=tuple(
                op_id
                for op_id in self.active_operations
                if self.operations[op_id].command.descriptor_id
                == descriptor.handle.descriptor_id
            ),
            gpu_ready_tokens=gpu_hit,
            recoverable_tokens=recoverable,
            cpu_standalone_tokens=min(limit, cpu_hit) if cpu_hit is not None else None,
            gpu_resident_blocks_by_group={
                g: len(blocks) for g, blocks in gpu_counts.items()
            },
            cpu_ready_blocks_by_group=dict(cpu_counts),
            lookup_state="COMPLETE" if recoverable is not None else "PENDING",
            lookup_basis=(
                "native coordinator + CPUOffloadingSpec lookup; completed writes only"
            ),
            reuse_basis="DESCRIPTOR_ONLY"
            if next_tokens is None
            else "ASSUMED_CONTINUATION",
            next_prompt_tokens=next_tokens,
            count_basis=count_basis,
            prefill_without_restore_tokens=max(0, next_tokens - gpu_hit)
            if next_tokens is not None
            else None,
            prefill_after_restore_tokens=max(0, next_tokens - recoverable)
            if next_tokens is not None and recoverable is not None
            else None,
            grace=self.grace_observation(descriptor),
            gpu_retention_bytes=len(
                {
                    allocation
                    for allocations in gpu_counts.values()
                    for allocation in allocations
                }
            )
            * self.gpu_allocation_bytes,
            offload_target_tokens=offload_target,
            offload_object_bytes=sum(
                descriptor.objects[key].num_bytes for key in offload_keys
            ),
        )

    def poll(self) -> None:
        now = time.monotonic()
        while self.deadlines and self.deadlines[0][0] <= now:
            deadline, kind, descriptor_id = heapq.heappop(self.deadlines)
            if kind == "OPERATION":
                if descriptor_id in self.active_operations:
                    heapq.heappush(
                        self.deadlines,
                        (now + self.config.metadata_ttl_seconds, kind, descriptor_id),
                    )
                elif (
                    operation := self.operations.pop(descriptor_id, None)
                ) is not None:
                    self.idempotency.pop(
                        (
                            operation.command.owner_scope,
                            operation.command.idempotency_key,
                        ),
                        None,
                    )
                    self.action_ids.pop(
                        (operation.command.owner_scope, operation.command.action_id),
                        None,
                    )
                continue
            if kind == "BINDING":
                entries = self.bindings.get(descriptor_id, {})
                if all(
                    did is not None and did not in self.descriptors
                    for did in entries.values()
                ):
                    self.bindings.pop(descriptor_id, None)
                    self.binding_expected.pop(descriptor_id, None)
                continue
            descriptor = self.descriptors.get(descriptor_id)
            if descriptor is None:
                continue
            if kind == "GRACE":
                grace = descriptor.grace
                if grace is not None and grace.state is GraceState.ACTIVE:
                    self.metrics["grace_expiry_delay_seconds"] += now - deadline
                    self.metrics["grace_expiry_delay_seconds_max"] = max(
                        self.metrics["grace_expiry_delay_seconds_max"], now - deadline
                    )
                    self.metrics["grace_expired_count"] += 1
                    self._end_grace(descriptor, GraceState.EXPIRED)
            elif kind == "METADATA":
                self._expire_descriptor(descriptor)
        while (
            self.events
            and self.events[0]["observed_at_monotonic"]
            < now - self.config.metadata_ttl_seconds
        ):
            self.events.popleft()
        self._progress_operations()
        self._refresh_dirty()
        if self.residency_changed:
            self.residency_changed = False
            self._event("KV_STATE_CHANGED")

    def _expire_descriptor(self, descriptor: Descriptor) -> None:
        did = descriptor.handle.descriptor_id
        # Metadata may outlive its public TTL to finish DMA or deferred DROP.
        # That must not extend a KEEP demand on shared GPU allocations.
        self._policy_changed(descriptor)
        grace = descriptor.grace
        if grace is not None and grace.state is GraceState.ACTIVE:
            heapq.heappush(self.deadlines, (grace.deadline, "METADATA", did))
            return
        if any(
            self.operations[op_id].command.descriptor_id == did
            for op_id in self.active_operations
        ):
            heapq.heappush(
                self.deadlines,
                (time.monotonic() + self.config.metadata_ttl_seconds, "METADATA", did),
            )
            return
        self._retry_drop(descriptor)
        if self.drop_gpu.get(did) or self.drop_cpu.get(did):
            # Public queries still expire on time. Retain only the existing
            # cleanup state until its references/fences allow reclamation.
            heapq.heappush(
                self.deadlines,
                (time.monotonic() + self.config.metadata_ttl_seconds, "METADATA", did),
            )
            return
        self._end_grace(descriptor, GraceState.EXPIRED)
        self._policy_changed(descriptor)
        for obj in descriptor.objects.values():
            self.by_cpu_key[obj.key].discard(did)
            if not self.by_cpu_key[obj.key]:
                del self.by_cpu_key[obj.key]
        gpu_hashes = {
            key for obj in descriptor.objects.values() for key in obj.gpu_hashes
        }
        for key in gpu_hashes:
            self.by_gpu_hash[key].discard(did)
            if not self.by_gpu_hash[key]:
                del self.by_gpu_hash[key]
        self.drop_gpu.pop(did, None)
        self.drop_cpu.pop(did, None)
        self.offload_gpu.pop(did, None)
        self._event("DESCRIPTOR_EXPIRED", did)
        self.descriptors.pop(did)
        self.cpu_backed.pop(did, None)
        # Deleted descriptors cannot drive a later dirty refresh of these blocks.
        for key in gpu_hashes:
            for block in self.gpu_replicas(key):
                self._index_free(block)
        for allocation in descriptor.observed_allocations:
            self.by_allocation[allocation].discard(did)
            if not self.by_allocation[allocation]:
                del self.by_allocation[allocation]
        binding = descriptor.handle.binding
        line = (binding.owner_scope, binding.job_id, binding.line_id)
        self.by_line[line].discard(did)
        if not self.by_line[line]:
            del self.by_line[line]
            if line not in self.active_lines:
                self.line_calls.pop(line, None)
        heapq.heappush(
            self.deadlines,
            (
                time.monotonic() + self.config.metadata_ttl_seconds,
                "BINDING",
                self._binding_key(binding),
            ),
        )

    def invalidate(self) -> None:
        """Release only GRACE. Native reset still drains its own transfer jobs."""
        for descriptor in self.descriptors.values():
            descriptor.invalidated = True
            self._end_grace(descriptor, GraceState.INVALIDATED)
            self._policy_changed(descriptor)
        self.drop_gpu.clear()
        self.drop_cpu.clear()
        self.offload_gpu.clear()
        self.identity = self.identity.model_copy(
            update={"engine_epoch": uuid.uuid4().hex}
        )
        self._event("ENGINE_RESET")

    def _target(
        self, descriptor: Descriptor, command: PolicyRequest
    ) -> tuple[int, frozenset[OffloadKey]]:
        configs = self.connector.config.kv_group_configs
        alignment = (
            self.connector.config.tokens_per_hash
            if self.connector.config.supports_partial_tail
            else math.lcm(
                *(
                    g.tokens_per_chunk
                    if command.action is Action.OFFLOAD
                    else g.tokens_per_block
                    for g in configs
                )
            )
        )
        target = command.target_resume_token_end
        if target is None:
            limit = (
                descriptor.handle.candidate_token_limit
                if command.action is Action.OFFLOAD
                else descriptor.handle.computed_token_limit
            )
            target = limit // alignment * alignment
        if target > descriptor.handle.computed_token_limit or target % alignment:
            raise KVControlError("INVALID_RESUME_POINT")
        if command.action is not Action.OFFLOAD:
            return target, frozenset(
                key
                for key, obj in descriptor.objects.items()
                if obj.token_end <= target
            )
        keys = set()
        for group in configs:
            ends = (
                [target]
                if group.requires_cow_source
                else list(
                    range(group.tokens_per_chunk, target + 1, group.tokens_per_chunk)
                )
            )
            if not group.requires_cow_source and target % group.tokens_per_chunk:
                ends.append(target)
            for end in ends:
                if end:
                    key = make_offload_key(
                        descriptor.hashes[
                            end // self.connector.config.tokens_per_hash - 1
                        ],
                        group.group_idx,
                    )
                    if key not in descriptor.objects:
                        raise KVControlError("INVALID_RESUME_MANIFEST")
                    keys.add(key)
        return target, frozenset(keys)

    def _policy_keys(self, descriptor: Descriptor) -> frozenset[OffloadKey]:
        if (
            descriptor.policy is None
            or descriptor.invalidated
            or time.monotonic() >= descriptor.handle.expires_at_monotonic
        ):
            return frozenset()
        return self._target(descriptor, descriptor.policy)[1]

    def _policy_changed(self, descriptor: Descriptor) -> None:
        """Shared DROP intents also depend on other owners' soft demand."""
        self.dirty_descriptors.add(descriptor.handle.descriptor_id)
        for obj in descriptor.objects.values():
            self.dirty_descriptors.update(self.by_cpu_key.get(obj.key, ()))
            for gpu_hash in obj.gpu_hashes:
                self.dirty_descriptors.update(self.by_gpu_hash.get(gpu_hash, ()))

    def _handoff(self, descriptor: Descriptor, keys: Collection[OffloadKey]) -> int:
        grace = descriptor.grace
        if grace is None:
            return 0
        grace.handed_off.update(
            gpu_hash for key in keys for gpu_hash in descriptor.objects[key].gpu_hashes
        )
        released = {
            allocation
            for allocation, required in grace.required_hashes.items()
            if required <= grace.handed_off
        }
        held = [
            grace.held[allocation][0]
            for allocation in released
            if allocation in grace.held
        ]
        self._end_grace(descriptor, GraceState.HANDED_OFF, released)
        return sum(block.ref_cnt == 0 for block in held) * self.gpu_allocation_bytes

    def _receipt(
        self,
        command: PolicyRequest,
        descriptor: Descriptor,
        operation_id: str,
        status: ActionStatus,
        target: int,
        skipped: dict[str, int],
        committed: int = 0,
        reused: int = 0,
        error: str | None = None,
        reclaimed: int = 0,
    ) -> ActionReceipt:
        observation = self._observe(descriptor)
        manifest = f"{descriptor.handle.state_manifest_ref}:{target}"
        return ActionReceipt(
            action_id=command.action_id,
            idempotency_key=command.idempotency_key,
            action=command.action,
            descriptor_id=command.descriptor_id,
            owner_scope=command.owner_scope,
            engine_epoch=self.identity.engine_epoch,
            status=status,
            operation_id=operation_id,
            applied_policy_version=descriptor.policy_version,
            state_version=self.state_version,
            event_seq=self.event_seq,
            requested_manifest_ref=manifest,
            accepted_manifest_ref=manifest + ":accepted:" + operation_id
            if status not in (ActionStatus.STALE, ActionStatus.UNSUPPORTED)
            else None,
            completed_manifest_ref=manifest + ":completed:" + operation_id
            if status in (ActionStatus.APPLIED, ActionStatus.PARTIAL)
            else None,
            target_resume_token_end=target,
            gpu_ready_tokens=observation.gpu_ready_tokens,
            recoverable_tokens=observation.recoverable_tokens,
            cpu_standalone_tokens=observation.cpu_standalone_tokens,
            cpu_committed_bytes=committed,
            cpu_reused_bytes=reused,
            gpu_reclaimed_bytes=reclaimed,
            skipped_reasons=skipped,
            error=error,
            grace=observation.grace,
        )

    def apply(self, command: PolicyRequest) -> ActionReceipt:
        self.poll()
        if command.expected_engine_epoch != self.identity.engine_epoch:
            raise KVControlError("STALE_ENGINE_EPOCH")
        key = (command.owner_scope, command.idempotency_key)
        previous = self.idempotency.get(key)
        if previous is not None:
            operation = self.operations[previous]
            if operation.command != command:
                raise KVControlError("IDEMPOTENCY_CONFLICT")
            return operation.receipt
        if (command.owner_scope, command.action_id) in self.action_ids:
            raise KVControlError("ACTION_ID_CONFLICT")
        descriptor = self._get(
            command.descriptor_id, command.owner_scope, command.expected_engine_epoch
        )
        target, target_keys = self._target(descriptor, command)
        operation_id = uuid.uuid4().hex
        binding = descriptor.handle.binding
        stale = (
            descriptor.invalidated
            or command.expected_policy_version != descriptor.policy_version
            or command.source_llm_call_id != binding.llm_call_id
            or command.expected_tail_request_id != binding.request_id
            or (
                descriptor.policy is not None
                and command.expected_tail_version
                < descriptor.policy.expected_tail_version
            )
        )
        unsupported = (
            command.action in (Action.KEEP, Action.OFFLOAD)
            and not self.config.retention_preferences
        )
        status = (
            ActionStatus.STALE
            if stale
            else ActionStatus.UNSUPPORTED
            if unsupported
            else ActionStatus.ACCEPTED
        )
        operation = Operation(
            command,
            self._receipt(command, descriptor, operation_id, status, target, {}),
            target_keys,
        )
        self.operations[operation_id] = operation
        self.idempotency[key] = operation_id
        self.action_ids[(command.owner_scope, command.action_id)] = operation_id
        heapq.heappush(
            self.deadlines,
            (
                time.monotonic() + self.config.metadata_ttl_seconds,
                "OPERATION",
                operation_id,
            ),
        )
        if stale or unsupported:
            return operation.receipt
        descriptor.policy = command
        descriptor.policy_version = command.policy_version
        did = descriptor.handle.descriptor_id
        self.drop_gpu.pop(did, None)
        self.drop_cpu.pop(did, None)
        self.offload_gpu.pop(did, None)
        self._policy_changed(descriptor)
        self._event("POLICY_APPLIED", did)
        skipped: Counter[str] = Counter()
        reclaimed = 0
        if command.action is Action.KEEP:
            # Preference is visible before the protection reference is removed.
            self._refresh_dirty()
            self._handoff(descriptor, target_keys)
            status = ActionStatus.APPLIED
        elif command.action is Action.DROP:
            self._capture_drop(descriptor, target_keys)
            reclaimed = self._handoff(descriptor, target_keys)
            skipped.update(self._retry_drop(descriptor))
            status = ActionStatus.PARTIAL if skipped else ActionStatus.APPLIED
        else:
            for obj in descriptor.objects.values():
                if obj.token_end > target:
                    continue
                for gpu_hash in obj.gpu_hashes:
                    for block in self.gpu_replicas(gpu_hash):
                        allocation = (block.block_id, block.allocation_generation)
                        operation.gpu_targets[allocation] = (
                            block.cache_mapping_generation
                        )
                        self.by_allocation[allocation].add(did)
                        descriptor.observed_allocations.add(allocation)
            sources = []
            for obj_key in target_keys:
                obj = descriptor.objects[obj_key]
                state = self.cpu.replica_state(obj.key)
                if state is not None and state[2]:
                    operation.reused_bytes += obj.num_bytes
                    self._handoff(descriptor, [obj.key])
                    continue
                blocks = []
                for gpu_hash in obj.gpu_hashes:
                    replicas = self.gpu_replicas(gpu_hash)
                    if replicas:
                        blocks.append(replicas[0])
                # PENDING is observed even if the original source already left
                # the GPU index after a native flush. Its job owns completion.
                if state is None and len(blocks) != len(obj.gpu_hashes):
                    skipped["SOURCE_LOST"] += 1
                    continue
                sources.append(
                    DescriptorStoreSource(
                        obj.key,
                        obj.group,
                        obj.token_end,
                        tuple(b.block_id for b in blocks),
                        tuple(b.allocation_generation for b in blocks),
                        obj.first_block_index,
                    )
                )
            outcomes = self.connector.enqueue_store_for_descriptor(
                sources,
                descriptor.context,
                operation_id,
                self.pool,
                descriptor.handle.candidate_token_limit,
            )
            handed_off = []
            for obj_key, outcome in outcomes.items():
                obj = descriptor.objects[obj_key]
                if outcome in ("ACCEPTED", "PENDING"):
                    jobs = self.connector._store_job_by_key[obj_key]
                    assert len(jobs) == 1
                    job_id = next(iter(jobs))
                    if outcome == "ACCEPTED":
                        operation.job_bytes[job_id] = (
                            operation.job_bytes.get(job_id, 0) + obj.num_bytes
                        )
                    else:
                        operation.pending_reuse_bytes[job_id] = (
                            operation.pending_reuse_bytes.get(job_id, 0) + obj.num_bytes
                        )
                    handed_off.append(obj_key)
                elif outcome == "READY":
                    operation.reused_bytes += obj.num_bytes
                    handed_off.append(obj_key)
                else:
                    skipped[outcome] += 1
            self._handoff(descriptor, handed_off)
            if not self.connector.operation_pending(operation_id):
                status = self._offload_terminal(descriptor, operation)
        operation.receipt = self._receipt(
            command,
            descriptor,
            operation_id,
            status,
            target,
            dict(skipped),
            reused=operation.reused_bytes,
            reclaimed=reclaimed,
        )
        if status is ActionStatus.ACCEPTED:
            self.active_operations.add(operation_id)
        elif command.action is Action.OFFLOAD and status is ActionStatus.APPLIED:
            self.offload_gpu[did] = operation_id
            self.dirty_descriptors.add(did)
        self._refresh_dirty()
        return operation.receipt

    def status(self, query: OperationQuery) -> ActionReceipt:
        self.poll()
        if query.expected_engine_epoch != self.identity.engine_epoch:
            raise KVControlError("STALE_ENGINE_EPOCH")
        operation = self.operations.get(query.operation_id)
        if operation is None:
            raise KVControlError("OPERATION_EXPIRED")
        if operation.command.owner_scope != query.owner_scope:
            raise KVControlError("OWNER_MISMATCH")
        return operation.receipt

    def _cpu_complete(
        self, descriptor: Descriptor, target: int, keys: Collection[OffloadKey]
    ) -> bool:
        if not all(
            (state := self.cpu.replica_state(key)) is not None and state[2]
            for key in keys
        ):
            return False
        prefix = HashPrefix(
            descriptor.handle.descriptor_id, descriptor.hashes, target + 1, target + 1
        )
        hit = self.connector.query_prefix(prefix, 0, descriptor.context)
        return hit is not None and hit >= target

    def _offload_terminal(
        self, descriptor: Descriptor, operation: Operation
    ) -> ActionStatus:
        target = self._target(descriptor, operation.command)[0]
        if self._cpu_complete(descriptor, target, operation.target_keys):
            return ActionStatus.APPLIED
        if any(
            (state := self.cpu.replica_state(key)) is not None and state[2]
            for key in operation.target_keys
        ):
            return ActionStatus.PARTIAL
        return ActionStatus.FAILED

    def _progress_operations(self) -> None:
        for operation_id in list(self.active_operations):
            operation = self.operations[operation_id]
            for job_id, (success, errors) in self.connector.take_store_outcomes(
                operation_id
            ).items():
                operation.errors.extend(errors)
                if success:
                    operation.committed_bytes += operation.job_bytes.pop(job_id, 0)
                    operation.reused_bytes += operation.pending_reuse_bytes.pop(
                        job_id, 0
                    )
            if self.connector.operation_pending(operation_id):
                continue
            descriptor = self.descriptors[operation.command.descriptor_id]
            status = self._offload_terminal(descriptor, operation)
            if descriptor.invalidated or descriptor.policy != operation.command:
                status = ActionStatus.STALE
            self._event("OPERATION_FINISHED", descriptor.handle.descriptor_id)
            operation.receipt = self._receipt(
                operation.command,
                descriptor,
                operation_id,
                status,
                operation.receipt.target_resume_token_end or 0,
                operation.receipt.skipped_reasons,
                committed=operation.committed_bytes,
                reused=operation.reused_bytes,
                error="; ".join(operation.errors) or None,
            )
            operation.receipt = operation.receipt.model_copy(
                update={"applied_policy_version": operation.command.policy_version}
            )
            self.active_operations.remove(operation_id)
            if status is ActionStatus.APPLIED:
                self.offload_gpu[descriptor.handle.descriptor_id] = operation_id
            self.dirty_descriptors.add(descriptor.handle.descriptor_id)

    def _retry_offload_gpu(self, descriptor: Descriptor) -> None:
        did = descriptor.handle.descriptor_id
        operation_id = self.offload_gpu.get(did)
        if operation_id is None:
            return
        operation = self.operations.get(operation_id)
        if (
            operation is None
            or descriptor.invalidated
            or descriptor.policy != operation.command
            or not self._policy_keys(descriptor)
        ):
            self.offload_gpu.pop(did, None)
            return
        target = self._target(descriptor, operation.command)[0]
        skipped: Counter[str] = Counter()
        if not self._cpu_complete(descriptor, target, operation.target_keys):
            skipped["CPU_BACKUP_UNAVAILABLE"] += len(operation.gpu_targets)
        else:
            for allocation, generation in list(operation.gpu_targets.items()):
                block = self.pool.blocks[allocation[0]]
                if (
                    block.allocation_generation != allocation[1]
                    or block.cache_mapping_generation != generation
                    or block.block_hash is None
                ):
                    del operation.gpu_targets[allocation]
                    continue
                hashes = self.pool.cached_block_hashes_by_block.get(
                    block.block_id, set()
                ) | {block.block_hash}
                if self._other_demand(descriptor, (), hashes, keep_only=True):
                    skipped["OTHER_KEEP"] += 1
                elif block.block_id in self.connector._block_id_to_pending_jobs:
                    skipped["IN_FLIGHT"] += 1
                elif block.ref_cnt:
                    skipped["SHARED_REQUEST_OR_GRACE"] += 1
                else:
                    result = self.pool.evict_if_idle(*allocation, generation)
                    if result == "APPLIED":
                        operation.gpu_reclaimed_bytes += self.gpu_allocation_bytes
                    if result in (
                        "APPLIED",
                        "ABSENT",
                        "STALE_ALLOCATION",
                        "STALE_MAPPING",
                    ):
                        del operation.gpu_targets[allocation]
                        self._index_free(block)
                    else:
                        skipped[result] += 1
        operation.receipt = self._receipt(
            operation.command,
            descriptor,
            operation_id,
            ActionStatus.APPLIED,
            target,
            dict(skipped),
            committed=operation.committed_bytes,
            reused=operation.reused_bytes,
            reclaimed=operation.gpu_reclaimed_bytes,
            error="; ".join(operation.errors) or None,
        )
        if not operation.gpu_targets:
            self.offload_gpu.pop(did, None)

    def _capture_drop(
        self, descriptor: Descriptor, keys: Collection[OffloadKey]
    ) -> None:
        did = descriptor.handle.descriptor_id
        gpu = self.drop_gpu[did] = {}
        cpu = self.drop_cpu[did] = {}
        for key in keys:
            obj = descriptor.objects[key]
            for gpu_hash in obj.gpu_hashes:
                for block in self.pool.cached_block_hash_to_block.get_all_blocks(
                    gpu_hash
                ):
                    allocation = (block.block_id, block.allocation_generation)
                    gpu[allocation] = block.cache_mapping_generation
                    self.by_allocation[allocation].add(did)
                    descriptor.observed_allocations.add(allocation)
            state = self.cpu.replica_state(key)
            if state is not None:
                cpu[key] = state[1]

    def _other_demand(
        self,
        descriptor: Descriptor,
        keys: Collection[OffloadKey],
        gpu_hashes: Collection[BlockHashWithGroupId] = (),
        *,
        keep_only: bool = False,
    ) -> bool:
        did = descriptor.handle.descriptor_id
        others: set[str] = set()
        for key in keys:
            others.update(self.by_cpu_key.get(key, ()))
        for gpu_hash in gpu_hashes:
            others.update(self.by_gpu_hash.get(gpu_hash, ()))
        for other_id in others - {did}:
            other = self.descriptors[other_id]
            if other.policy is None or other.policy.action is Action.DROP:
                continue
            if keep_only and other.policy.action is not Action.KEEP:
                continue
            policy_keys = self._policy_keys(other)
            if policy_keys.intersection(keys) or any(
                gpu_hash in gpu_hashes
                for key in policy_keys
                for gpu_hash in other.objects[key].gpu_hashes
            ):
                return True
        return False

    def _retry_drop(self, descriptor: Descriptor) -> dict[str, int]:
        did = descriptor.handle.descriptor_id
        skipped: Counter[str] = Counter()
        if (
            descriptor.invalidated
            or descriptor.policy is None
            or descriptor.policy.action is not Action.DROP
        ):
            self.drop_gpu.pop(did, None)
            self.drop_cpu.pop(did, None)
            return {}
        for allocation, mapping_generation in list(self.drop_gpu.get(did, {}).items()):
            block = self.pool.blocks[allocation[0]]
            if (
                block.allocation_generation != allocation[1]
                or block.cache_mapping_generation != mapping_generation
                or block.block_hash is None
            ):
                del self.drop_gpu[did][allocation]
                continue
            hashes = self.pool.cached_block_hashes_by_block.get(
                block.block_id, set()
            ) | {block.block_hash}
            if self._other_demand(descriptor, (), hashes):
                skipped["OTHER_DEMAND"] += 1
            elif block.block_id in self.connector._block_id_to_pending_jobs:
                skipped["IN_FLIGHT"] += 1
            elif block.ref_cnt:
                skipped["SHARED_REQUEST_OR_GRACE"] += 1
            else:
                result = self.pool.evict_if_idle(*allocation, mapping_generation)
                if result in ("APPLIED", "ABSENT", "STALE_ALLOCATION", "STALE_MAPPING"):
                    del self.drop_gpu[did][allocation]
                    self._index_free(block)
                else:
                    skipped[result] += 1
        for key, generation in list(self.drop_cpu.get(did, {}).items()):
            if self._other_demand(descriptor, [key]):
                skipped["OTHER_DEMAND"] += 1
                continue
            result = self.cpu.evict_if_idle(key, generation)
            if result in ("APPLIED", "ABSENT", "STALE_ALLOCATION"):
                del self.drop_cpu[did][key]
            else:
                skipped[result] += 1
        return dict(skipped)

    def _refresh_dirty(self) -> None:
        dirty, self.dirty_descriptors = self.dirty_descriptors, set()
        if not dirty:
            return
        started = time.monotonic()
        for did in dirty:
            descriptor = self.descriptors.get(did)
            if descriptor is None:
                continue
            grace = descriptor.grace
            if (
                grace is not None
                and grace.state is GraceState.ACTIVE
                and any(block.block_hash is None for block, _ in grace.held.values())
            ):
                self._end_grace(descriptor, GraceState.INVALIDATED)
            policy = descriptor.policy
            keys = self._policy_keys(descriptor)
            self.cpu_backed[did] = bool(
                policy is not None
                and not descriptor.invalidated
                and policy.action is Action.OFFLOAD
                and self._cpu_complete(
                    descriptor, self._target(descriptor, policy)[0], keys
                )
            )
            self._retry_offload_gpu(descriptor)
            if did in self.drop_gpu or did in self.drop_cpu:
                before = (
                    len(self.drop_gpu.get(did, ())),
                    len(self.drop_cpu.get(did, ())),
                )
                self._retry_drop(descriptor)
                after = (
                    len(self.drop_gpu.get(did, ())),
                    len(self.drop_cpu.get(did, ())),
                )
                if after != before:
                    self._event("DROP_PROGRESS", did)
                if (
                    after == (0, 0)
                    and descriptor.handle.expires_at_monotonic <= time.monotonic()
                ):
                    self._expire_descriptor(descriptor)
                    continue
            for obj in descriptor.objects.values():
                for key in obj.gpu_hashes:
                    for block in self.gpu_replicas(key):
                        if block.ref_cnt == 0:
                            self._index_free(block)
        self.metrics["preference_update_count"] += len(dirty)
        self.metrics["preference_update_seconds"] += time.monotonic() - started

    def _priority(self, block: KVCacheBlock) -> int:
        if block.block_hash is None:
            return -1
        keys = self.pool.cached_block_hashes_by_block.get(block.block_id, set()) | {
            block.block_hash
        }
        priority = 2
        for gpu_hash in keys:
            for did in self.by_gpu_hash.get(gpu_hash, ()):
                descriptor = self.descriptors[did]
                policy = descriptor.policy
                if policy is None or descriptor.invalidated:
                    continue
                if not any(
                    gpu_hash in descriptor.objects[key].gpu_hashes
                    for key in self._policy_keys(descriptor)
                ):
                    continue
                if policy.action is Action.KEEP:
                    return 3
        return priority

    def _index_free(self, block: KVCacheBlock) -> None:
        if not self.config.retention_preferences or block.ref_cnt or block.is_null:
            return
        self.free_age += 1
        self.free_versions[block.block_id] = self.free_age
        age = self.free_recency.setdefault(block.block_id, self.free_age)
        heapq.heappush(
            self.free_heap,
            (
                self._priority(block),
                age,
                block.block_id,
                block.allocation_generation,
                self.free_age,
            ),
        )

    def _select_free(self, count: int) -> list[KVCacheBlock]:
        started = time.monotonic()
        self._refresh_dirty()
        result: list[KVCacheBlock] = []
        while len(result) < count:
            priority, _, bid, generation, version = heapq.heappop(self.free_heap)
            block = self.pool.blocks[bid]
            if (
                block.ref_cnt
                or block.allocation_generation != generation
                or self.free_versions.get(bid) != version
            ):
                continue
            if priority != self._priority(block):
                self._index_free(block)
                continue
            self.pool.free_block_queue.remove(block)
            del self.free_versions[bid]
            del self.free_recency[bid]
            result.append(block)
        # Amortized cleanup of superseded preference entries, independent of
        # physical capacity; no per-step scan of the GPU block pool.
        if len(self.free_heap) > 2 * len(self.free_versions) + 1:
            self.free_heap = [
                entry
                for entry in self.free_heap
                if self.free_versions.get(entry[2]) == entry[4]
            ]
            heapq.heapify(self.free_heap)
        self.metrics["free_selection_count"] += 1
        self.metrics["free_selection_seconds"] += time.monotonic() - started
        return result
