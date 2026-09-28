# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical reference and protocol behavior using native pools, without CUDA."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_kv_cache_config,
    _make_mamba_hybrid_kv_cache_config,
    _make_vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
    _make_partial_tail_scheduler,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_control.manager import BINDING_KEY, KVControlError, KVControlManager
from vllm.v1.kv_control.protocol import (
    ActionStatus,
    CallBinding,
    GraceState,
    KVControlConfig,
    OperationQuery,
    PolicyRequest,
    QueryRequest,
    TelemetryRequest,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def control(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("vllm.v1.kv_control.manager.time.monotonic", lambda: clock[0])
    config = _make_mamba_hybrid_kv_cache_config()
    config.num_blocks = 16
    cache = KVCacheManager(
        config, max_model_len=128, scheduler_block_size=16, hash_block_size=4
    )
    connector = _make_partial_tail_scheduler()
    connector.manager = CPUOffloadingManager(16, store_threshold=0)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            world_size=1,
            tensor_parallel_size=1,
        ),
        speculative_config=None,
        model_config=SimpleNamespace(model="unit", revision="1", dtype="float32"),
        kv_transfer_config=SimpleNamespace(engine_id="unit"),
        compute_hash=lambda: "config",
    )
    manager = KVControlManager(
        KVControlConfig(enabled=True, finish_grace_ttl_ms=250),
        vllm_config,
        cache,
        connector,
    )
    return manager, clock


@pytest.fixture
def full_attention_control(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("vllm.v1.kv_control.manager.time.monotonic", lambda: clock[0])
    config = _make_kv_cache_config()
    cache = KVCacheManager(
        config, max_model_len=128, scheduler_block_size=16, hash_block_size=16
    )
    vllm_config = _make_vllm_config(extra_config={"blocks_per_chunk": 2})
    vllm_config.speculative_config = None
    vllm_config.model_config.revision = "1"
    vllm_config.compute_hash.return_value = "config"
    connector = OffloadingConnectorScheduler(
        MockOffloadingSpec(build_offloading_config(vllm_config, config)),
        vllm_config,
        config,
    )
    connector.manager = CPUOffloadingManager(16, store_threshold=0)
    manager = KVControlManager(
        KVControlConfig(enabled=True, finish_grace_ttl_ms=250),
        vllm_config,
        cache,
        connector,
    )
    return manager, clock


def finish(
    control, name="call", line="line", hashes=None, before_finish=None, computed=32
):
    manager, _ = control
    hash_tokens = manager.pool.hash_block_size
    hashes = hashes or [
        BlockHash(f"h{i}".encode()) for i in range(computed // hash_tokens)
    ]
    binding = CallBinding(
        owner_scope="scope",
        job_id="job",
        line_id=line,
        request_id=name,
        llm_call_id=name,
        attempt=0,
        context_epoch=0,
    )
    request = SimpleNamespace(
        request_id=name,
        kv_transfer_params={BINDING_KEY: binding.model_dump()},
        num_computed_tokens=computed,
        num_in_flight_tokens=0,
        num_tokens=computed + 1,
        num_prompt_tokens=computed,
        block_hashes=hashes,
        status=RequestStatus.FINISHED_STOPPED,
    )
    manager.on_request(request)
    blocks = []
    for group in manager.connector.config.kv_group_configs:
        for end in range(group.tokens_per_block, computed + 1, group.tokens_per_block):
            block = manager.pool.get_new_blocks(1)[0]
            manager.pool._insert_block_hash(
                make_block_hash_with_group_id(
                    hashes[end // hash_tokens - 1], group.group_idx
                ),
                block,
                end,
            )
            blocks.append(block)
    if before_finish is not None:
        before_finish(request)
    metadata = manager.finish(request)
    manager.pool.free_blocks(blocks)
    descriptor = manager.descriptors[metadata["descriptor"]["descriptor_id"]]
    return descriptor, blocks


def query(manager, descriptor):
    return manager.query(
        QueryRequest(
            descriptor_id=descriptor.handle.descriptor_id,
            owner_scope="scope",
            expected_engine_epoch=manager.identity.engine_epoch,
        )
    )


def command(manager, descriptor, action, version=1, **kwargs):
    return PolicyRequest(
        action_id=f"a-{descriptor.handle.descriptor_id}-{version}",
        idempotency_key=f"k-{descriptor.handle.descriptor_id}-{version}",
        action=action,
        descriptor_id=descriptor.handle.descriptor_id,
        owner_scope="scope",
        expected_engine_epoch=manager.identity.engine_epoch,
        expected_policy_version=version - 1,
        policy_version=version,
        source_llm_call_id=descriptor.handle.binding.llm_call_id,
        expected_tail_request_id=descriptor.handle.binding.request_id,
        expected_tail_version=1,
        decision_ref="unit",
        **kwargs,
    )


def test_grace_pins_real_blocks_and_expires_without_deleting_descriptor(control):
    manager, clock = control
    descriptor, blocks = finish(control)
    assert all(b.ref_cnt == 1 for b in blocks)
    free = manager.pool.get_num_free_blocks()
    deadline = descriptor.grace.deadline
    assert query(manager, descriptor).gpu_ready_tokens == 32
    assert query(manager, descriptor).prefill_without_restore_tokens is None
    assert descriptor.grace.deadline == deadline
    assert manager.pool.get_num_free_blocks() == free
    clock[0] += 0.251
    manager.poll()
    assert all(b.ref_cnt == 0 for b in blocks)
    assert descriptor.grace.state is GraceState.EXPIRED
    assert query(manager, descriptor).gpu_ready_tokens == 32


def test_keep_handoff_is_idempotent_and_does_not_pin(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    cmd = command(manager, descriptor, "KEEP")
    receipt = manager.apply(cmd)
    assert receipt.status is ActionStatus.APPLIED
    assert descriptor.grace.state is GraceState.HANDED_OFF
    assert all(b.ref_cnt == 0 for b in blocks)
    assert manager.apply(cmd) == receipt
    with pytest.raises(KVControlError, match="IDEMPOTENCY_CONFLICT"):
        manager.apply(cmd.model_copy(update={"decision_ref": "changed"}))
    allocated = manager.pool.get_new_blocks(manager.pool.get_num_free_blocks())
    assert len(allocated) == 15
    assert query(manager, descriptor).gpu_ready_tokens == 0


def test_drop_waits_for_shared_request_then_rechecks_original_allocation(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    manager.pool.touch([blocks[0]])
    receipt = manager.apply(command(manager, descriptor, "DROP"))
    assert receipt.status is ActionStatus.PARTIAL
    assert receipt.gpu_reclaimed_bytes == 0
    assert blocks[0].block_hash is not None
    manager.pool.free_blocks([blocks[0]])
    manager.poll()
    assert blocks[0].block_hash is None
    status = manager.status(
        OperationQuery(
            operation_id=receipt.operation_id,
            owner_scope="scope",
            expected_engine_epoch=manager.identity.engine_epoch,
        )
    )
    assert status.status is ActionStatus.PARTIAL  # Preserve the initial snapshot.


@pytest.mark.parametrize("computed", [16, 48])
def test_grace_includes_gpu_tail_smaller_than_cpu_chunk(
    full_attention_control, computed
):
    manager, _ = full_attention_control
    descriptor, blocks = finish(full_attention_control, computed=computed)
    observation = query(manager, descriptor)
    assert observation.gpu_ready_tokens == computed
    assert observation.gpu_resident_blocks_by_group == {0: computed // 16}
    assert all(block.ref_cnt == 1 for block in blocks)


@pytest.mark.parametrize("action", ["KEEP", "DROP"])
@pytest.mark.parametrize("target", [None, 16])
def test_gpu_policy_range_is_independent_of_cpu_chunks(
    full_attention_control, action, target
):
    manager, _ = full_attention_control
    descriptor, blocks = finish(full_attention_control, computed=48)
    receipt = manager.apply(
        command(manager, descriptor, action, target_resume_token_end=target)
    )
    assert receipt.status is ActionStatus.APPLIED
    assert receipt.target_resume_token_end == (target or 48)
    selected = (target or 48) // 16
    assert all(block.ref_cnt == 0 for block in blocks[:selected])
    assert all(block.ref_cnt == 1 for block in blocks[selected:])
    assert all(
        (block.block_hash is None) == (action == "DROP") for block in blocks[:selected]
    )


def test_offload_hands_off_complete_chunks_and_preserves_gpu_tail(
    full_attention_control,
):
    manager, _ = full_attention_control
    descriptor, blocks = finish(full_attention_control, computed=48)
    receipt = manager.apply(command(manager, descriptor, "OFFLOAD"))
    assert receipt.status is ActionStatus.ACCEPTED
    assert receipt.target_resume_token_end == 32
    assert [block.ref_cnt for block in blocks] == [1, 1, 1]
    jobs = list(manager.connector._jobs)
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={
                    job: manager.connector.config.num_workers for job in jobs
                }
            )
        )
    )
    manager.poll()
    assert [block.ref_cnt for block in blocks] == [0, 0, 1]
    assert query(manager, descriptor).cpu_standalone_tokens == 32


def test_gpu_tail_drop_respects_another_lines_full_chunk_offload(
    full_attention_control,
):
    manager, _ = full_attention_control
    backed, _ = finish(full_attention_control, name="backed", line="backed")
    manager.apply(command(manager, backed, "OFFLOAD"))
    jobs = list(manager.connector._jobs)
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={
                    job: manager.connector.config.num_workers for job in jobs
                }
            )
        )
    )
    dropped, blocks = finish(
        full_attention_control, name="dropped", line="dropped", computed=16
    )
    receipt = manager.apply(command(manager, dropped, "DROP"))
    assert receipt.status is ActionStatus.PARTIAL
    assert receipt.skipped_reasons["OTHER_DEMAND"]
    assert blocks[0].block_hash is not None
    manager.apply(command(manager, backed, "DROP", version=2))
    manager.poll()
    assert blocks[0].block_hash is None


@pytest.mark.parametrize("tier", ["gpu", "cpu"])
def test_deferred_drop_survives_metadata_expiry(control, tier):
    manager, clock = control
    descriptor, blocks = finish(control)
    key = next(iter(descriptor.objects))
    if tier == "gpu":
        manager.pool.touch([blocks[0]])
    else:
        manager.cpu.prepare_store([key], descriptor.context)
        manager.cpu.complete_store([key], descriptor.context)
        manager.cpu.prepare_load([key], descriptor.context)
    assert (
        manager.apply(command(manager, descriptor, "DROP")).status
        is ActionStatus.PARTIAL
    )
    clock[0] += 301
    manager.poll()
    with pytest.raises(KVControlError, match="DESCRIPTOR_EXPIRED"):
        query(manager, descriptor)
    if tier == "gpu":
        assert blocks[0].block_hash is not None
        manager.pool.free_blocks([blocks[0]])
    else:
        assert manager.cpu.replica_state(key) is not None
        manager.cpu.complete_load([key], descriptor.context)
    manager.poll()
    assert blocks[0].block_hash is None
    assert manager.cpu.replica_state(key) is None
    assert not manager.descriptors
    assert not manager.drop_gpu and not manager.drop_cpu
    assert (
        not manager.by_gpu_hash and not manager.by_cpu_key and not manager.by_allocation
    )


def test_expired_keep_returns_to_normal_lru_order(control):
    manager, clock = control
    descriptor, kept = finish(control)
    manager.apply(command(manager, descriptor, "KEEP"))
    allocated = manager.pool.get_new_blocks(
        manager.pool.get_num_free_blocks() - len(kept)
    )
    clock[0] += 301
    manager.poll()
    newer = allocated.pop()
    manager.pool._insert_block_hash(
        make_block_hash_with_group_id(BlockHash(b"newer"), 0), newer, 16
    )
    manager.pool.free_blocks([newer])
    assert manager.pool.get_new_blocks(1)[0] in kept


@pytest.mark.parametrize("retained_by_group", [(0, 0), (32, 0), (32, 16)])
def test_offload_pins_sources_until_dma_completion_then_plain_query_sees_cpu(
    control, retained_by_group
):
    manager, clock = control
    descriptor, blocks = finish(control)
    held = [
        block
        for index, block in enumerate(blocks)
        if (index % 2 + 1) * 16 <= retained_by_group[index // 2]
    ]
    manager.pool.touch(held)
    receipt = manager.apply(command(manager, descriptor, "OFFLOAD"))
    assert receipt.status is ActionStatus.ACCEPTED
    clock[0] += 1
    manager.poll()
    jobs = list(manager.connector._jobs)
    assert jobs
    assert any(b.ref_cnt for b in blocks)
    completed = OffloadingWorkerMetadata(
        completed_jobs={job: manager.connector.config.num_workers for job in jobs}
    )
    manager.connector.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=completed)
    )
    manager.poll()
    receipt = manager.status(
        OperationQuery(
            operation_id=receipt.operation_id,
            owner_scope="scope",
            expected_engine_epoch=manager.identity.engine_epoch,
        )
    )
    assert receipt.status is ActionStatus.APPLIED
    assert receipt.cpu_committed_bytes > 0
    assert all(b.ref_cnt == (1 if b in held else 0) for b in blocks)
    assert all(b.block_hash is None for b in blocks if b not in held)
    observation = query(manager, descriptor)
    assert observation.gpu_ready_tokens == min(retained_by_group)
    assert observation.cpu_standalone_tokens == 32
    assert observation.recoverable_tokens == 32
    assert observation.cpu_extension_tokens == 32 - min(retained_by_group)
    assert observation.reuse_basis == "DESCRIPTOR_ONLY"

    request = Request("successor", list(range(33)), SamplingParams(max_tokens=1), None)
    request.block_hashes = descriptor.hashes.copy()
    connector = OffloadingConnector.__new__(OffloadingConnector)
    connector.connector_scheduler = manager.connector
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.connector = connector
    scheduler.kv_cache_manager = manager.cache
    _, local, _, divergent = scheduler._get_local_prefix_cache_hit(request)
    assert not divergent
    connector.on_new_request(request)
    external, _ = connector.get_num_new_matched_tokens(request, local)
    assert external is not None
    assert observation.gpu_ready_tokens == local
    assert observation.recoverable_tokens == local + external
    manager.pool.free_blocks(held)
    manager.poll()
    assert query(manager, descriptor).gpu_ready_tokens == 0
    assert query(manager, descriptor).cpu_standalone_tokens == 32


def test_offload_gpu_reclamation_waits_for_shared_keep_then_rechecks(control):
    manager, clock = control
    kept, _ = finish(control, "kept", "keeper")
    manager.apply(command(manager, kept, "KEEP"))
    backed, blocks = finish(control, "backed", "offloader")
    receipt = manager.apply(command(manager, backed, "OFFLOAD"))
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={
                    job: manager.connector.config.num_workers
                    for job in manager.connector._jobs
                }
            )
        )
    )
    manager.poll()
    assert all(b.block_hash is not None for b in blocks)
    assert manager.operations[receipt.operation_id].receipt.skipped_reasons[
        "OTHER_KEEP"
    ]
    clock[0] += 0.251
    manager.poll()
    assert all(b.block_hash is not None for b in blocks)
    successor = kept.handle.binding.model_copy(
        update={"llm_call_id": "next", "request_id": "next"}
    )
    manager.on_request(
        SimpleNamespace(
            request_id="next", kv_transfer_params={BINDING_KEY: successor.model_dump()}
        )
    )
    manager.poll()
    assert all(b.block_hash is None for b in blocks)
    assert query(manager, backed).cpu_standalone_tokens == 32
    assert backed.handle.descriptor_id not in manager.offload_gpu


def test_expired_keep_does_not_survive_metadata_held_by_older_dma(control):
    manager, clock = control
    descriptor, blocks = finish(control)
    manager.apply(command(manager, descriptor, "OFFLOAD"))
    manager.apply(command(manager, descriptor, "KEEP", version=2))
    assert manager._priority(blocks[0]) == 3
    clock[0] += 301
    manager.poll()
    assert descriptor.handle.descriptor_id in manager.descriptors
    assert manager._priority(blocks[0]) == 2


def test_target_query_validates_new_content_without_registering_or_touching(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    target = Request("query", list(range(33)), SamplingParams(max_tokens=1), None)
    target.block_hashes = descriptor.hashes.copy()
    before = (
        [b.ref_cnt for b in blocks],
        len(manager.descriptors),
        dict(manager.line_calls),
    )
    observed = manager.query_target(target, manager.identity.engine_epoch, "q")
    assert observed.reuse_basis == "TARGET_REQUEST"
    assert observed.prompt_tokens == 33 and observed.gpu_ready_tokens == 32
    target.block_hashes = [BlockHash(b"different")] * len(target.block_hashes)
    cold = manager.query_target(target, manager.identity.engine_epoch, "q")
    assert cold.gpu_ready_tokens == cold.recoverable_tokens == 0
    assert before == (
        [b.ref_cnt for b in blocks],
        len(manager.descriptors),
        dict(manager.line_calls),
    )


def test_failed_offload_never_reclaims_gpu_sources(control):
    manager, clock = control
    descriptor, blocks = finish(control)
    receipt = manager.apply(command(manager, descriptor, "OFFLOAD"))
    jobs = list(manager.connector._jobs)
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={j: manager.connector.config.num_workers for j in jobs},
                failed_jobs={j: ["worker copy failed"] for j in jobs},
            )
        )
    )
    clock[0] += 1
    manager.poll()
    assert (
        manager.operations[receipt.operation_id].receipt.status is ActionStatus.FAILED
    )
    assert all(b.block_hash is not None for b in blocks)
    assert descriptor.handle.descriptor_id not in manager.offload_gpu


def test_uncompleted_cow_index_is_not_published_as_ready(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    manager.pending_writes[(blocks[-1].block_id, blocks[-1].allocation_generation)] = 1
    assert query(manager, descriptor).gpu_ready_tokens == 16
    manager.complete_step(SimpleNamespace(kv_control_step=1))
    # Only registered completion fences can publish a write, not a query.
    assert query(manager, descriptor).gpu_ready_tokens == 16


def test_metadata_expiry_reclaims_reverse_indexes_and_receipts(control):
    manager, clock = control
    descriptor, _ = finish(control)
    binding = descriptor.handle.binding
    manager.apply(command(manager, descriptor, "KEEP"))
    clock[0] += 301
    manager.poll()
    assert manager.resolve(binding).status == "DESCRIPTOR_EXPIRED"
    assert not manager.descriptors
    assert not manager.by_cpu_key
    assert not manager.by_gpu_hash
    assert not manager.by_allocation
    assert not manager.operations
    assert not manager.idempotency
    clock[0] += 301
    manager.poll()
    assert not manager.bindings
    assert not manager.binding_expected


def test_resolve_reports_engine_clock_without_renewing_descriptor(control):
    manager, clock = control
    descriptor, _ = finish(control)
    binding = descriptor.handle.binding
    first = manager.resolve(binding)
    assert first.observed_at_monotonic == clock[0]
    clock[0] += 42
    again = manager.resolve(binding)
    assert again.observed_at_monotonic == clock[0]
    assert again.descriptors[0].expires_at_monotonic == (
        first.descriptors[0].expires_at_monotonic
    )
    assert (
        again.descriptors[0].expires_at_monotonic - again.observed_at_monotonic
        == manager.config.metadata_ttl_seconds - 42
    )
    clock[0] += manager.config.metadata_ttl_seconds
    expired = manager.resolve(binding)
    assert expired.status == "DESCRIPTOR_EXPIRED"
    assert expired.observed_at_monotonic == clock[0]


def test_known_prompt_logprobs_disable_reuse_without_erasing_residency(control):
    manager, _ = control
    descriptor, _ = finish(control)
    observation = manager.query(
        QueryRequest(
            descriptor_id=descriptor.handle.descriptor_id,
            owner_scope="scope",
            expected_engine_epoch=manager.identity.engine_epoch,
            next_prompt_tokens=40,
            count_basis="EXACT",
            requests_prompt_logprobs=True,
        )
    )
    assert observation.gpu_ready_tokens == observation.recoverable_tokens == 0
    assert observation.prefill_after_restore_tokens == 40
    assert sum(observation.gpu_resident_blocks_by_group.values()) == 4


def test_metadata_ttl_cannot_shorten_an_active_grace(control):
    manager, clock = control
    manager.config = manager.config.model_copy(update={"metadata_ttl_seconds": 0.1})
    descriptor, blocks = finish(control)
    clock[0] += 0.15
    manager.poll()
    with pytest.raises(KVControlError, match="DESCRIPTOR_EXPIRED"):
        query(manager, descriptor)
    assert all(block.ref_cnt == 1 for block in blocks)
    clock[0] += 0.11
    manager.poll()
    assert all(block.ref_cnt == 0 for block in blocks)


def test_cpu_eviction_revokes_complete_bundle_preference(control):
    manager, _ = control
    descriptor, _ = finish(control)
    receipt = manager.apply(command(manager, descriptor, "OFFLOAD"))
    jobs = list(manager.connector._jobs)
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={
                    job: manager.connector.config.num_workers for job in jobs
                }
            )
        )
    )
    manager.poll()
    did = descriptor.handle.descriptor_id
    assert manager.cpu_backed[did]
    key = next(iter(manager.operations[receipt.operation_id].target_keys))
    state = manager.cpu.replica_state(key)
    manager.cpu.evict_if_idle(key, state[1])
    manager.poll()
    assert not manager.cpu_backed[did]


def test_unimplemented_preference_mode_preserves_grace_on_rejection(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    manager.config = manager.config.model_copy(update={"retention_preferences": False})
    receipt = manager.apply(command(manager, descriptor, "KEEP"))
    assert receipt.status is ActionStatus.UNSUPPORTED
    assert descriptor.grace.state is GraceState.ACTIVE
    assert all(b.ref_cnt == 1 for b in blocks)


@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("expired", [False, True])
def test_deferred_drop_cannot_revoke_a_new_hash_mapping(control, replace, expired):
    manager, clock = control
    descriptor, blocks = finish(control)
    block = blocks[0]
    manager.pool.touch([block])
    assert (
        manager.apply(command(manager, descriptor, "DROP")).status
        is ActionStatus.PARTIAL
    )
    if expired:
        clock[0] += 301
        manager.poll()
    if replace:
        manager.pool._remove_cached_block_hashes(block)
    new_hash = make_block_hash_with_group_id(BlockHash(b"new-content"), 0)
    manager.pool._insert_block_hash(new_hash, block, 16)
    manager.pool.free_blocks([block])
    manager.poll()
    assert manager.pool.cached_block_hash_to_block.contain(new_hash, block.block_id)


def test_completed_write_advances_observation_watermark(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    block = blocks[-1]
    allocation = (block.block_id, block.allocation_generation)
    manager.pending_writes[allocation] = 1
    manager.writes_by_step[1].add(allocation)
    before = query(manager, descriptor)
    assert before.gpu_ready_tokens == 16
    manager.complete_step(SimpleNamespace(kv_control_step=1))
    after = query(manager, descriptor)
    assert after.gpu_ready_tokens == 32
    assert after.state_version > before.state_version
    assert after.event_seq > before.event_seq


def test_partial_handoff_preserves_the_other_grace_references(control):
    manager, clock = control
    descriptor, blocks = finish(control)
    manager.apply(command(manager, descriptor, "KEEP", target_resume_token_end=16))
    assert [block.ref_cnt for block in blocks] == [0, 1, 0, 1]
    clock[0] += 0.251
    manager.poll()
    assert all(block.ref_cnt == 0 for block in blocks)


def test_old_finish_cannot_reestablish_policy_or_grace_after_successor(control):
    manager, _ = control

    def start_successor(request):
        raw = dict(request.kv_transfer_params[BINDING_KEY])
        raw.update(llm_call_id="successor", request_id="successor")
        manager.on_request(
            SimpleNamespace(
                request_id="successor", kv_transfer_params={BINDING_KEY: raw}
            )
        )

    descriptor, blocks = finish(control, before_finish=start_successor)
    assert descriptor.grace is None
    assert (
        manager.apply(command(manager, descriptor, "KEEP")).status is ActionStatus.STALE
    )
    assert all(block.ref_cnt == 0 for block in blocks)


def test_invalidated_shared_keep_wakes_deferred_drop(control):
    manager, _ = control
    kept, _ = finish(control, "kept", "keeper")
    manager.apply(command(manager, kept, "KEEP"))
    dropped, blocks = finish(control, "dropped", "dropper")
    receipt = manager.apply(command(manager, dropped, "DROP"))
    assert receipt.status is ActionStatus.PARTIAL
    assert receipt.skipped_reasons["OTHER_DEMAND"]
    raw = kept.handle.binding.model_copy(
        update={"llm_call_id": "next", "request_id": "next"}
    )
    manager.on_request(
        SimpleNamespace(
            request_id="next", kv_transfer_params={BINDING_KEY: raw.model_dump()}
        )
    )
    manager.poll()
    assert all(block.block_hash is None for block in blocks)


@pytest.mark.parametrize(
    "next_tokens,hit", [(0, 0), (16, 12), (17, 16), (32, 28), (33, 32), (100, 32)]
)
def test_next_request_length_limits_consumable_prefix(control, next_tokens, hit):
    manager, _ = control
    descriptor, _ = finish(control)
    observed = manager.query(
        QueryRequest(
            descriptor_id=descriptor.handle.descriptor_id,
            owner_scope="scope",
            expected_engine_epoch=manager.identity.engine_epoch,
            next_prompt_tokens=next_tokens,
            count_basis="EXACT",
        )
    )
    # The native hybrid coordinator may require an earlier checkpoint.
    assert observed.gpu_ready_tokens <= hit
    assert (
        observed.prefill_without_restore_tokens
        == next_tokens - observed.gpu_ready_tokens
    )


def test_expired_event_history_reports_a_watermark_gap(control):
    manager, clock = control
    manager._event("ENGINE_RESET")
    clock[0] += 301
    result = manager.telemetry(
        TelemetryRequest(
            owner_scope="scope",
            expected_engine_epoch=manager.identity.engine_epoch,
            after_event_seq=0,
        )
    )
    assert result.events == ()
    assert result.events_gap


def test_policy_replacement_cannot_be_overwritten_by_old_copy_completion(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    pending = manager.apply(command(manager, descriptor, "OFFLOAD"))
    kept = manager.apply(command(manager, descriptor, "KEEP", version=2))
    jobs = list(manager.connector._jobs)
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={
                    job: manager.connector.config.num_workers for job in jobs
                }
            )
        )
    )
    manager.poll()
    assert manager.operations[pending.operation_id].receipt.status is ActionStatus.STALE
    assert manager.operations[kept.operation_id].receipt.status is ActionStatus.APPLIED
    assert descriptor.policy.action == "KEEP"
    assert all(block.ref_cnt == 0 for block in blocks)


def test_idle_core_uses_deadline_without_scheduling_a_gpu_step():
    import queue
    from unittest.mock import MagicMock

    from vllm.v1.engine.core import EngineCoreProc

    core = MagicMock()
    core.has_work.return_value = False
    core.is_running.side_effect = [True, False]
    core.process_input_queue_block = True
    core.input_queue.empty.return_value = True
    core.input_queue.get.side_effect = queue.Empty
    core.scheduler.kv_control.next_timeout.return_value = 0.25
    EngineCoreProc._process_input_queue(core)
    core.input_queue.get.assert_called_once_with(block=True, timeout=0.25)
    assert core.scheduler.kv_control.poll.call_count == 2
    core.step_fn.assert_not_called()


def test_reset_fences_old_copy_and_ignores_its_late_completion(control):
    manager, _ = control
    descriptor, blocks = finish(control)
    manager.apply(command(manager, descriptor, "OFFLOAD"))
    jobs = set(manager.connector._jobs)
    old_epoch = manager.identity.engine_epoch
    manager.invalidate()
    manager.connector.reset_cache()
    assert manager.connector._current_batch_jobs_to_flush == jobs
    assert all(block.ref_cnt == 0 for block in blocks)
    allocated = manager.pool.get_new_blocks(manager.pool.get_num_free_blocks())
    refs = [block.ref_cnt for block in allocated]
    manager.connector.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={
                    job: manager.connector.config.num_workers for job in jobs
                }
            )
        )
    )
    manager.poll()
    assert [block.ref_cnt for block in allocated] == refs
    assert not manager.cpu._policy.blocks
    assert manager.resolve(descriptor.handle.binding).status == "DESCRIPTOR_EXPIRED"
    with pytest.raises(KVControlError, match="STALE_ENGINE_EPOCH"):
        manager.query(
            QueryRequest(
                descriptor_id=descriptor.handle.descriptor_id,
                owner_scope="scope",
                expected_engine_epoch=old_epoch,
            )
        )


def test_gpu_hole_is_not_replaced_by_remaining_block_count(control):
    manager, clock = control
    descriptor, blocks = finish(control)
    clock[0] += 0.251
    manager.poll()
    # FA's first block is missing; later attention and Mamba blocks remain.
    manager.pool.evict_if_idle(blocks[0].block_id, blocks[0].allocation_generation)
    observed = query(manager, descriptor)
    assert observed.gpu_ready_tokens == observed.recoverable_tokens == 0
    assert sum(observed.gpu_resident_blocks_by_group.values()) == 3
    assert observed.prefix_uncovered_tokens == 33
