# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attach actual engine/choice identities before serializing inference ingress."""

from copy import copy

from vllm.sampling_params import SamplingParams
from vllm.v1.kv_control.protocol import CallBinding


def bind_choice(
    params: SamplingParams, parent_id: str, index: int, count: int
) -> SamplingParams:
    extra = params.extra_args or {}
    transfer = extra.get("kv_transfer_params") or {}
    if "kv_control_binding" not in transfer:
        return params
    CallBinding.model_validate(transfer["kv_control_binding"])
    result = copy(params)
    transfer = dict(transfer)
    transfer["kv_control_child"] = dict(
        parent_engine_request_id=parent_id,
        output_index=index,
        output_count=count,
        **transfer.pop("kv_control_input", {"input_index": 0, "input_count": 1}),
    )
    result.extra_args = dict(extra, kv_transfer_params=transfer)
    return result


def bind_input(params: SamplingParams, index: int, count: int) -> SamplingParams:
    extra = params.extra_args or {}
    transfer = extra.get("kv_transfer_params") or {}
    if "kv_control_binding" not in transfer:
        return params
    result = copy(params)
    transfer = dict(transfer)
    transfer["kv_control_input"] = dict(input_index=index, input_count=count)
    result.extra_args = dict(extra, kv_transfer_params=transfer)
    return result
