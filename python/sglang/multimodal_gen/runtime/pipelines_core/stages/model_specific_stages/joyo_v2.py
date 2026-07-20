# SPDX-License-Identifier: Apache-2.0
"""JoyO V2 pipeline utilities.

Pure helpers for input preparation, shared by pipeline stages and standalone
tests. Ported from `diffusers.pipelines.joyo_v2.pipeline_joyo_v2`.

- build_position_ids: 3D MRoPE positions for the packed [text, pixel] sequence.
- safe_devide_append: pad text tokens so the full packed sequence is divisible
  by an SP denominator (default 8). Mirrors Joytron's data prep contract.
"""

from __future__ import annotations

import torch

DEFAULT_SP_DENOMINATOR: int = 8
QWEN3_VL_PAD_TOKEN_ID: int = 151643


def build_position_ids(
    t: int,
    h: int,
    w: int,
    device: torch.device,
    pre_len: int = 0,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Build 3D position ids for MRoPE over a flattened pixel volume.

    Args:
        t, h, w: latent dimensions after patchify.
        device: target device.
        pre_len: text sequence length offset. Joytron continues the pixel
            temporal axis after text, so pixel time index starts at `pre_len`.
        dtype: dtype of the returned tensor (int32 matches the transformer's
            rotary buffer indexer).
    Returns:
        (3, t*h*w) tensor with [temporal, height, width] positions.
    """
    t_ids = (
        torch.arange(pre_len, pre_len + t, device=device, dtype=dtype)
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(t, h, w)
    )
    h_ids = (
        torch.arange(h, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(2)
        .expand(t, h, w)
    )
    w_ids = (
        torch.arange(w, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(1)
        .expand(t, h, w)
    )
    return torch.stack([t_ids.flatten(), h_ids.flatten(), w_ids.flatten()], dim=0)


def build_packed_position_ids(
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Build MRoPE position ids for the full packed [text, pixel] sequence.

    Text tokens occupy positions [0, text_len) on all 3 axes. Pixel tokens
    follow with the temporal axis continuing at `text_len`.

    Returns:
        (3, text_len + latent_t*latent_h*latent_w)
    """
    text_pos = (
        torch.arange(text_len, device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(3, -1)
    )
    pixel_pos = build_position_ids(
        latent_t, latent_h, latent_w, device, pre_len=text_len, dtype=dtype
    )
    return torch.cat([text_pos, pixel_pos], dim=1)


def compute_safe_devide_pad(
    valid_lens: torch.Tensor,
    num_pixels_per_sample: int,
    denominator: int = DEFAULT_SP_DENOMINATOR,
) -> tuple[int, int]:
    """Compute how many text pad tokens to append so the packed sequence is
    divisible by `denominator`.

    The total packed length is `sum(valid_lens) + num_pixels_per_sample * num_samples`
    (each sample contributes valid text tokens plus a fixed pixel block).
    This function returns the pad length to add to a single sample, and the
    index of that sample (the shortest — matches Joytron's convention).

    Args:
        valid_lens: (num_samples,) int tensor, valid token count per sample.
        num_pixels_per_sample: pixel token count per sample.
        denominator: SP world size (usually 8).
    Returns:
        (pad_len, min_idx). `pad_len == 0` means no padding needed.
    """
    num_samples = valid_lens.numel()
    extra_lens = (num_pixels_per_sample * num_samples) % denominator
    total_valid = int(valid_lens.sum().item())
    remainder = (total_valid + extra_lens) % denominator
    if remainder == 0:
        return 0, 0
    pad_len = denominator - remainder
    min_idx = int(torch.argmin(valid_lens).item())
    return pad_len, min_idx


def safe_devide_append(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_pixels_per_sample: int,
    denominator: int = DEFAULT_SP_DENOMINATOR,
    pad_token_id: int = QWEN3_VL_PAD_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place-style pad input_ids/attention_mask so the packed sequence is
    divisible by `denominator`.

    Returns modified (input_ids, attention_mask). If the batch is already
    aligned, the inputs are returned unchanged.

    The pad is applied to the *shortest* sample in the batch, appended after
    its valid tokens. If the batch's max seq_len is too short to hold the
    padding, both tensors are right-padded first (with zeros / mask=0), then
    the shortest sample's tail is filled with `pad_token_id` and mask=1.
    """
    valid_lens = attention_mask.sum(dim=1)
    pad_len, min_idx = compute_safe_devide_pad(
        valid_lens, num_pixels_per_sample, denominator
    )
    if pad_len == 0:
        return input_ids, attention_mask

    min_len = int(valid_lens[min_idx].item())
    seq_len = input_ids.size(1)
    batch_size = input_ids.size(0)

    if min_len + pad_len > seq_len:
        expand_len = min_len + pad_len - seq_len
        input_ids = torch.cat(
            [
                input_ids,
                torch.zeros(
                    (batch_size, expand_len), dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.zeros(
                    (batch_size, expand_len), dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

    input_ids[min_idx, min_len : min_len + pad_len] = pad_token_id
    attention_mask[min_idx, min_len : min_len + pad_len] = 1
    return input_ids, attention_mask
