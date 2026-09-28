# Experimental engine KV control (schema 1)

This extension registers metadata for completed generation requests and exposes
read-only prefix observations plus `KEEP`, `OFFLOAD`, and `DROP`. It is disabled
by default. The engine owns all storage, allocation, copies, and recovery.
There is no RESTORE command. Submit the next inference request normally, even
when its observed prefix exists only in CPU memory.

The implementation targets vLLM commit
`98dff2a81d747d1dba01a47f939f48c3526d4206` (0.29.0). Its internal connector and
pool interfaces must be revalidated when upgrading.

## Configuration and supported layouts

Use the normal multiprocess EngineCore with APC enabled and
`OffloadingConnector + CPUOffloadingSpec`. Full-attention and Mamba `align`
groups are supported with the native noncanonical CPU layout. PP, DP, DCP and
PCP must each be 1; speculative decoding and other group types are not
implemented by this control layer. Unsupported enabled configurations fail
explicitly at initialization. Disabled configurations retain native behavior.
The synchronous in-process EngineCore has no autonomous idle loop and cannot
enable this extension; do not set `VLLM_ENABLE_V1_MULTIPROCESSING=0`.

```json
{
  "kv_control": {
    "enabled": true,
    "finish_grace_ttl_ms": 250,
    "metadata_ttl_seconds": 300,
    "retention_preferences": true
  }
}
```

Pass this object as `additional_config` (CLI: `--additional-config`). A TTL of
zero disables physical finish holds. `retention_preferences=false` keeps
registration, GRACE, queries, and safe DROP; KEEP and explicit OFFLOAD return
UNSUPPORTED. Native automatic CPU offload remains independent of this switch.

Configure the connector, for example, as:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "CPUOffloadingSpec",
    "cpu_bytes_to_use": 1073741824,
    "store_threshold": 0,
    "offload_prompt_only": false
  }
}
```

The native default for `offload_prompt_only` is true; decode state is not
implicitly stored just because a descriptor covers it. Both automatic and
explicit stores honor the completed-token watermark and `max_offload_tokens`,
including zero. An uncomputed final sampled token is never an offload candidate.

The local Qwen3.5-9B acceptance baseline uses BF16, TP=4 on GPU 0–3, Mamba
`align`, and **`prefix_cache_retention_interval=None`** for dense checkpoints.
Set the actual Python configuration value to `None`; do not treat the string
`"None"` as equivalent. The default sparse value 0 can omit the Mamba checkpoint
at the 1056-token boundary while attention objects exist. A zero CPU-recoverable
prefix in that case is a real observation. The control layer does not synthesize
checkpoints. TP=4 is the verified deployment point, not a claim that other
layouts have been exercised.

250 ms is an experimental handoff window. It is not a production recommendation.

## Request identity and completion delivery

Attach this metadata to the ordinary Chat Completions, Completions, or Responses
request (with an OpenAI SDK, use `extra_body`):

```json
{
  "kv_transfer_params": {
    "kv_control_binding": {
      "schema_version": 1,
      "owner_scope": "gateway-deployment",
      "job_id": "job-1",
      "line_id": "line-1",
      "request_id": "request-1",
      "llm_call_id": "call-1",
      "attempt": 0,
      "context_epoch": 0
    }
  }
}
```

The inference ingress records the actual engine parent, input index and output
choice, including `n>1`. The `kv_control_child` and `kv_control_input` keys are
reserved and rejected in external API requests. A binding is independent of
the HTTP response ID. A subsequent call on the same owner/job/line invalidates
older retention policies; an older request finishing later cannot reestablish
them. The engine does not replace a gateway's authoritative tail/version check.

Finish metadata is carried in `kv_transfer_params.kv_control_v1`. Use
`/v1/kv/resolve` with the original binding after receiving a final response or
SSE terminal; this also works when a particular response format omits transfer
metadata. Resolve returns PENDING, READY, DESCRIPTOR_EXPIRED, or UNKNOWN_BINDING
and the actual child descriptors. It never returns generated text.

Resolve also returns `observed_at_monotonic` in the same engine clock domain as
each descriptor's `expires_at_monotonic`. Clients compute remaining TTL by
subtracting these fields, then anchor it at their local RPC start time so transport
latency cannot extend the deadline. Do not compare absolute monotonic clocks across
hosts. Repeated resolves retain the original engine expiry.

FlowPilot uses this sample for local source timers and also consumes matching
`DESCRIPTOR_EXPIRED` events. Timers work without descriptor queries or successful
control RPCs; an event-log gap does not require scanning all tails to expire local
references. Local cleanup sends no DROP and does not interrupt engine-owned DMA or
deferred reclamation. This client revision requires the resolve clock field;
missing fields are reported as resolution validation errors, never an invented TTL.

Descriptors retain hashes and object manifests, not a Request or prompt.
The reported `prefix_token_count`, `computed_token_limit`, and
`candidate_token_limit` have different meanings. Content identity includes the
native hash chain and model/layout compatibility digest. Descriptor identity
survives eviction, CPU movement and DROP until metadata expiry; reset changes
the engine epoch and invalidates old handles.

## Finite control API

All mutations run serially on the EngineCore thread through the existing
utility channel. API routes do not access workers, storage, or block maps.
The routes inherit normal vLLM `/v1` authentication. This is an interface for
a trusted gateway within one inference domain: `owner_scope` separates records
and validates commands, but is not an independently authenticated tenant
credential. Deployment authorization remains the gateway/server operator's
responsibility.

| Method | Route | Typed request/result |
| --- | --- | --- |
| GET | `/v1/kv/capabilities` | `CapabilitySet` |
| POST | `/v1/kv/resolve` | `CallBinding` / `ResolveResult` |
| POST | `/v1/kv/query` | `QueryRequest` / `PrefixObservation` |
| POST | `/v1/kv/apply` | `PolicyRequest` / `ActionReceipt` |
| POST | `/v1/kv/status` | `OperationQuery` / `ActionReceipt` |
| POST | `/v1/kv/telemetry` | `TelemetryRequest` / `ControlTelemetry` |

The exact frozen, extra-field-rejecting Pydantic contract is
[`protocol.py`](../../vllm/v1/kv_control/protocol.py). The independent
[`KVControlClient`](../../vllm/v1/kv_control/client.py) accepts a caller-owned
`httpx.AsyncClient`. It does not import FlowPilot or implement request admission.

Capabilities separately describe descriptor lookup, GRACE, preferences, CPU
store, native CPU reuse, transfer measurement, and cost estimates. Restore-cost
estimates and continuation proofs are currently **unsupported**. Native transfer
metrics are measurements; worker-summed transfer time is not wall-clock copy
latency or an internal restore-queue ETA.

Disabled control returns HTTP 501. Owner mismatch returns 403, expired metadata
410, and stale epoch or command conflicts 409. An action's FAILED/PARTIAL receipt
is an ordinary successful transport response; its nullable `error` field is
distinct from the transport error envelope.

## Queries and observations

```python
import httpx
from vllm.v1.kv_control.client import KVControlClient
from vllm.v1.kv_control.protocol import QueryRequest

async with httpx.AsyncClient(base_url="http://127.0.0.1:8029") as http:
    control = KVControlClient(http)
    capabilities = await control.capabilities()
    observation = await control.query(QueryRequest(
        descriptor_id=descriptor_id,
        owner_scope="gateway-deployment",
        expected_engine_epoch=capabilities.engine.engine_epoch,
    ))
```

Queries reuse native GPU coordinator and CPU connector matching rules. They do
not tokenize, touch LRU order, acquire references, prepare a request or copy KV.
Only completed GPU writes and CPU stores count as READY. Hybrid recovery needs
the required attention range and compatible Mamba checkpoint, with native
chunk/alignment restrictions. Physical object counts are reported separately
from consumable prefix length; disconnected fragments cannot inflate a prefix.
The supported OffloadingConnector starts CPU lookup at the common GPU hit across
all required groups, as ordinary inference does. A deeper full-attention hit
alone cannot advance that starting point when a recurrent group is missing.

`gpu_ready_tokens` is H_gpu, `recoverable_tokens` is H_all, and
`cpu_standalone_tokens` is the native CPU-only prefix. Pending CPU lookup returns
unknown recovery fields rather than a fabricated complete range. The response
also reports H_all−H_gpu and C−H_all when known, effective policy, in-flight
operations, state/event versions, and remaining GRACE.

An ID-only query uses `DESCRIPTOR_ONLY` and returns null next-request prefill
counts. Supplying `next_prompt_tokens` with an EXACT or ESTIMATED `count_basis`
produces `ASSUMED_CONTINUATION`, constrained by N−1 and native recovery boundaries.
It does not prove the next request's content. Known prompt-logprob or
skip-prefix-cache options suppress consumable hits while retaining physical
residency facts. Ordinary inference always revalidates actual input.
Re-rendering conversation history can change its token prefix even when the
assistant text is preserved verbatim; this does not shorten the old descriptor's
own available prefix. Physical eviction alone does not expire its metadata.

## Action and lifetime semantics

Commands include owner, engine epoch, source call, expected tail request/version,
expected/current policy versions, action ID and idempotency key. A repeated
identical command returns the same operation and committed receipt; reusing a
key for different content fails. Retry the same command after a transport timeout.

* GRACE is established before releasing the request's references, using deduped
  physical allocation references. Queries and retries never renew its deadline.
  An idle EngineCore wakes at the nearest deadline without submitting GPU work;
  an active engine releases at its next safe point and reports expiry delay.
* KEEP registers a soft preference, then releases the corresponding GRACE.
  It owns no lasting protection references and remains reclaimable under pressure.
* OFFLOAD reuses READY CPU objects and joins native PENDING stores. Missing objects
  use native allocation and copy jobs even after the Request was destroyed.
  Copy references and fences are acquired before GRACE handoff, and survive its
  TTL. All necessary workers must finish before READY is published. A recoverable
  failure is accepted only after DMA quiesces; fatal CUDA errors still fail the
  engine. Once the complete CPU target is READY, reclaim its original GPU mappings
  when they have no active references, copy protection, or other valid KEEP demand.
  Protected allocations retry after native/policy events, with allocation and
  mapping generation checks. CPU readiness is rechecked before reclamation. This
  makes blocks reusable inside the KV pool; it does not cudaFree the pool.
* DROP removes only this owner's demand/GRACE. It checks other demand, requests,
  GRACE, compute and transfer use. Deferred intents retry on relevant native and
  policy events, using original allocation and hash-mapping generations. A newly
  cached replica is not chased by an old DROP. Initial PARTIAL receipts remain
  historical snapshots; later cleanup is visible in query/events.

Subset actions leave other holds on their original deadline. Rejected, STALE
and UNSUPPORTED actions do not hand off protection. Integrity invalidation and
reset revoke affected guarantees explicitly. Metadata expiry does not shorten
an already active physical hold; internal records remain until safety work ends,
while public lookup can already report expired metadata.

ACCEPTED means asynchronous work exists. OFFLOAD reports APPLIED only if the
**original target** is CPU recoverable at completion. Objects evicted during the
operation cause PARTIAL/FAILED even when every new DMA succeeded. Receipts
separate newly committed bytes from reused bytes and actual newly allocatable
GPU capacity. Historical success does not promise current residency.

Telemetry contains only metadata, aggregate capacity/cost counters and a
time-bounded event log. `events_gap=true` requires resynchronization through
queries; events are not a complete authoritative cache inventory.

## Target input queries and offline costs

`POST /v1/kv/query-target` accepts `owner_scope`, `expected_engine_epoch`,
`query_id`, `api_kind` (`chat` or `responses`), and the original JSON `payload`.
Native serving renders the input; the input processor and EngineCore block hasher
produce the same target hashes used by inference. Each input returns its prompt
length, GPU/recoverable prefix, candidate CPU object bytes, engine identity and
state version with `reuse_basis=TARGET_REQUEST`. It does not register a descriptor,
submit inference, acquire KV references, or initiate restore. Cache-skip and prompt
logprobs requests report no reusable prefix. Observations can become stale before
ordinary inference acquires blocks; this is not a residency lease.

Capabilities expose `target_prefix_query` and `offload_gpu_reclaim` separately.
The engine does not manufacture time estimates. FlowPilot may combine native
object bytes with matching offline D2H/H2D and context-bucket prefill calibrations.
These are conditional costs, not engine scheduling or recovery commands.

Descriptors remain immutable per completed request/output branch. A line's current
tail points to its current descriptor; a successor call invalidates older demand.
Metadata expiry revokes KEEP even when physical safety work delays record removal.
GPU eviction and DROP alone do not remove descriptor metadata before its TTL.

## Validation scope

The native fixes apply to all groups in performance comparisons: completed-store
boundaries, partial-tail caps, and common-prefix membership independent of
protection references. Control is separately enabled per experiment.

Local evidence covers Qwen3.5-9B TP=4 dense checkpoint boundary cases, ordinary
CPU restoration versus recomputation, actual cascade dispatch with live GRACE,
idle expiry, delayed copies after CPU eviction, quiesced necessary-rank failure,
and Chat/Completions/Responses streaming correlation. The failure injection is at
the terminal copy result boundary; it is not a simulation of a recoverable GPU
device crash. Other parallel layouts and production workloads require their own
evidence. No FlowPilot/OpenHands implementation or workflow scheduling benefit
is included in this engine-only phase.
