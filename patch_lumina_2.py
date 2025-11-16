# A very basic ComfyUI plugin to patch Lumina 2 model, written by reakaakasky (https://civitai.com/user/reakaakasky).
# Put this file in the ComfyUI "custom_nodes" dir.

# 11/9/2025 v1, tested on ComfyUI v0.3.68. Nvidia card.

##### Settings #####

# Enable torch compile. Can +30% speed on newer Nivida hardware (RTX 3000+).
# Note: It needs to compile the model first time you enable and run it, usually takes 30~60s.
ENABLE_TORCH_COMPILE = False # True / False

# Lumina 2 can't run in fp16 mode, because some activation tensors in the middle of the model are overflowed.
# Enable this can let you run Lumina 2 in fp16 mode. It will recompute the layer in fp32 again if any overflow was detected.
# Note: You can use the built-in "ModelComputeDtype" node to force Comfyui using fp16 mode.
ENABLE_F32_FALLBACK = False

##### End of settings #####

from comfy.ldm.lumina.model import JointTransformerBlock
import torch
from typing import Optional
import logging

logging.info("patching Lumina 2 JointTransformerBlock")

if ENABLE_TORCH_COMPILE:
    JointTransformerBlock.forward = torch.compile(JointTransformerBlock.forward)

# Patch to support fp16
def forward_with_fp32_fallback(
    self,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: Optional[torch.Tensor] = None,
    transformer_options={},
):
    dtype = x.dtype
    out = self._forward(x, x_mask, freqs_cis, adaln_input, transformer_options)
    if x.dtype == torch.float16 and x.is_cuda:
        isinf, isnan = out.isinf().any(), out.isnan().any()
        if isinf or isnan:
            # print(f"inf {isinf}, nan {isnan}")
            with torch.amp.autocast_mode.autocast("cuda", torch.float32):
                out = self._forward(
                    x, x_mask, freqs_cis, adaln_input, transformer_options
                )
            # print(f"fixed out: dtype {out.dtype}, max {out.abs().max().item()}")
            out = out.to(dtype).nan_to_num()
    return out

if ENABLE_F32_FALLBACK:
    JointTransformerBlock._forward = JointTransformerBlock.forward
    JointTransformerBlock.forward = forward_with_fp32_fallback
