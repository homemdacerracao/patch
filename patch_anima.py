ENABLE_TORCH_COMPILE = False
ENABLE_F32_FALLBACK = True

try:
    from comfy.ldm.anima.model import TransformerBlock
except ImportError:
    TransformerBlock = None

import torch
import logging

if TransformerBlock is not None and ENABLE_TORCH_COMPILE:
    TransformerBlock.forward = torch.compile(TransformerBlock.forward)

def forward_with_fp32_fallback(
    self, 
    x, 
    context, 
    target_attention_mask=None, 
    source_attention_mask=None, 
    position_embeddings=None, 
    position_embeddings_context=None
):
    dtype = x.dtype
    
    out = self._original_forward(
        x, 
        context, 
        target_attention_mask, 
        source_attention_mask, 
        position_embeddings, 
        position_embeddings_context
    )
    
    if x.dtype == torch.float16 and x.is_cuda:
        isinf, isnan = out.isinf().any(), out.isnan().any()
        if isinf or isnan:
            with torch.amp.autocast_mode.autocast("cuda", torch.float32):
                out = self._original_forward(
                    x, 
                    context, 
                    target_attention_mask, 
                    source_attention_mask, 
                    position_embeddings, 
                    position_embeddings_context
                )
            out = out.to(dtype).nan_to_num()
            
    return out

if TransformerBlock is not None and ENABLE_F32_FALLBACK:
    if not hasattr(TransformerBlock, "_original_forward"):
        TransformerBlock._original_forward = TransformerBlock.forward
        TransformerBlock.forward = forward_with_fp32_fallback
