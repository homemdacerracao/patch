# A very basic ComfyUI plugin to patch Lumina 2 model, written by reakaakasky (https://civitai.com/user/reakaakasky).
# Put this file in the ComfyUI "custom_nodes" dir.

# 11/17/2025 v1.1, tested on ComfyUI v0.3.68. Nvidia card.
# Updated to support FP8 scaled models without errors.

##### Settings #####

# Enable torch compile. Can +30% speed on newer Nvidia hardware (RTX 3000+).
# Note: It needs to compile the model first time you enable and run it, usually takes 30~60s.
ENABLE_TORCH_COMPILE = False # True / False

# Lumina 2 can't run in fp16 mode, because some activation tensors in the middle of the model overflow.
# Enable this to run Lumina 2 in fp16 mode. It will recompute the layer in fp32 if overflow is detected.
# Note: You can use the built-in "ModelComputeDtype" node to force ComfyUI using fp16 mode.
ENABLE_F32_FALLBACK = False # True / False

# FP8 scaled models support.
# Enable this to properly handle FP8 (e.g., float8_e4m3fn) models by converting to fp16 for computation.
# This provides memory savings from FP8 storage with good fp16 performance, while maintaining fp32 fallback for stability.
ENABLE_FP8_SUPPORT = True # True / False

##### End of settings #####

from comfy.ldm.lumina.model import JointTransformerBlock
import torch
from typing import Optional
import logging

logging.info("Patching Lumina 2 JointTransformerBlock for FP16/FP8 support")

def is_fp8_dtype(dtype: torch.dtype) -> bool:
    """Check if the dtype is any FP8 variant."""
    fp8_dtypes = []
    if hasattr(torch, 'float8_e4m3fn'):
        fp8_dtypes.append(torch.float8_e4m3fn)
    if hasattr(torch, 'float8_e5m2'):
        fp8_dtypes.append(torch.float8_e5m2)
    if hasattr(torch, 'float8_e4m3fnuz'):
        fp8_dtypes.append(torch.float8_e4m3fnuz)
    if hasattr(torch, 'float8_e5m2fnuz'):
        fp8_dtypes.append(torch.float8_e5m2fnuz)
    return dtype in fp8_dtypes

if ENABLE_TORCH_COMPILE:
    logging.info("Torch compile enabled for JointTransformerBlock")
    JointTransformerBlock.forward = torch.compile(JointTransformerBlock.forward)

# Patch to support fp16 and fp8 with optional fallback
def forward_with_fp32_fallback(
    self,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: Optional[torch.Tensor] = None,
    transformer_options={},
):
    original_dtype = x.dtype
    original_device = x.device
    
    # Handle FP8: convert to fp16 for efficient computation
    compute_dtype = original_dtype
    if is_fp8_dtype(original_dtype) and ENABLE_FP8_SUPPORT:
        compute_dtype = torch.float16
        x = x.to(compute_dtype)
        if freqs_cis.dtype != compute_dtype:
            freqs_cis = freqs_cis.to(compute_dtype)
        if adaln_input is not None and adaln_input.dtype != compute_dtype:
            adaln_input = adaln_input.to(compute_dtype)
    
    # Run forward pass in compute dtype (fp16 for FP8 models, original dtype otherwise)
    out = self._forward(x, x_mask, freqs_cis, adaln_input, transformer_options)
    
    # Check for overflow and apply fp32 fallback if enabled
    if ENABLE_F32_FALLBACK and compute_dtype in [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]:
        if out.isinf().any() or out.isnan().any():
            logging.debug(f"Detected overflow in {compute_dtype}, recomputing in fp32")
            with torch.amp.autocast_mode.autocast(original_device.type, torch.float32):
                # Re-run with original inputs converted to fp32
                x_fp32 = x.to(torch.float32)
                freqs_cis_fp32 = freqs_cis.to(torch.float32)
                adaln_input_fp32 = adaln_input.to(torch.float32) if adaln_input is not None else None
                
                out = self._forward(x_fp32, x_mask, freqs_cis_fp32, adaln_input_fp32, transformer_options)
            
            # Convert back to compute dtype, then to original dtype
            out = out.to(compute_dtype).nan_to_num()
    
    # Convert final output back to original dtype (FP8 if it started as FP8)
    return out.to(original_dtype).nan_to_num()

# Apply patch if either feature is enabled
if ENABLE_F32_FALLBACK or ENABLE_FP8_SUPPORT:
    if not hasattr(JointTransformerBlock, '_forward'):
        JointTransformerBlock._forward = JointTransformerBlock.forward
    JointTransformerBlock.forward = forward_with_fp32_fallback
    logging.info(f"Forward patch applied (F32_FALLBACK: {ENABLE_F32_FALLBACK}, FP8_SUPPORT: {ENABLE_FP8_SUPPORT})")
