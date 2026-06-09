# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import torch


def build_uniform_expert_owner_maps(
    *,
    num_experts: int,
    device: torch.device | str,
    world_size: int | None = None,
    num_local_experts: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build expert owner/local-id tensors for contiguous uniform EP shards."""

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if world_size is not None and world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if num_local_experts is not None and num_local_experts <= 0:
        raise ValueError(
            f"num_local_experts must be positive, got {num_local_experts}"
        )
    if world_size is None and num_local_experts is None:
        raise ValueError("provide world_size or num_local_experts")

    if world_size is not None:
        if num_experts % world_size != 0:
            raise ValueError(
                f"num_experts {num_experts} must be divisible by world_size "
                f"{world_size}"
            )
        inferred_local = num_experts // world_size
        if num_local_experts is not None and num_local_experts != inferred_local:
            raise ValueError(
                f"num_local_experts {num_local_experts} does not match "
                f"num_experts/world_size {inferred_local}"
            )
        num_local_experts = inferred_local
    else:
        assert num_local_experts is not None
        if num_experts % num_local_experts != 0:
            raise ValueError(
                f"num_experts {num_experts} must be divisible by "
                f"num_local_experts {num_local_experts}"
            )

    experts = torch.arange(num_experts, dtype=torch.int32, device=device)
    return experts // num_local_experts, experts % num_local_experts


__all__ = ["build_uniform_expert_owner_maps"]
