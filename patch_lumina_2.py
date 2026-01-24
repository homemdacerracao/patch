# A very basic ComfyUI plugin to patch Lumina 2 based models to use fp16 "safely" on old GPUs.
# Lumina 2 based models means: Lumina 2, Z-Image, NewBie.
# Written by reakaakasky (https://civitai.com/user/reakaakasky).

# To clarify, this plugin isn't uploaded to GitHub because it's a very "dirty" plugin.
# "dirty" means it hot-patches/monkey patches the comfyui core code.
# This approach is terrible from a programming perspective.
# But this is the simplest approach I can think of.

# How to use:
# Put this file in the ComfyUI "custom_nodes" dir.
# Use "ModelComputeDtype" node and set dtype to "fp16".

# Version: v1 (1/6/2026)

##### Settings #####

# Main switch.
ENABLE_PATCH = True

# This patch can handle overflow in fp16 mode.
# It will automatically recompute the layer in fp32 again if overflow was detected.
# Enable this to print a debug log when overflow was detected, so you know it is working.
DEBUG_RECOMPUTE_PRINT = True

# Reduce some model weights **on the fly** during model loading to avoid overflow.
# Less fp32 recompute, faster speed. (e.g. It can eliminate all overflows in z-image, 80% in lumina 2)
#
# **Incompatible with LoRA**.
# Because the model weight changed after loading, if you load LoRA, you will get noise output.
# If you need LoRA, you can:
# - (fast) Disable this setting, then merge LoRA into a checkpoint and save it in advance, then enable this setting.
# - (slow) Disable this setting, and use LoRA.
REDUCE_WEIGHT = True

# Do NOT change this if you don't know what it is.
REDUCE_WEIGHT_DIV_FACTOR = 32

# Enable torch compile, maybe faster? Not tested on old GPUs.
ENABLE_TORCH_COMPILE = False

##### End of settings #####

from typing import Optional
import torch
import logging
import comfy.ldm.lumina.model as lumina

logger = logging.getLogger(__name__)
logger.info("[lunima fp16 patch] patch loaded")

NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}


def weight_scale_hook(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
) -> None:
    def scalable_weight(name: str) -> bool:
        for kw in ("qkv", "w3"):
            if kw in name:
                return True
        return False

    for key, w in state_dict.items():
        if key.endswith(".weight") and scalable_weight(key):
            base_name = key.removesuffix(".weight")

            # handle scaled fp8 model
            ws = state_dict.get(base_name + ".weight_scale", None)
            if ws is not None:
                ws.div_(other=REDUCE_WEIGHT_DIV_FACTOR)
                logger.info(f"reduced weight scale {key} {ws.dtype}")
            elif w.dtype in (torch.float32, torch.float16, torch.bfloat16):
                w.div_(REDUCE_WEIGHT_DIV_FACTOR)
                logger.info(f"reduced weight {key} {w.dtype}")
            else:
                logger.info(f"skipping {key}, unsupported dtype {w.dtype}")


# Patch JointTransformerBlock to support fp16
def forward_with_fp32_fallback(
    self,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: Optional[torch.Tensor] = None,
    transformer_options={},
):
    __tmp_module_name = getattr(self, "__tmp_module_name", "unknow")

    fallback_mode = "__fp16_fallback_mode" in transformer_options
    if fallback_mode:
        with torch.amp.autocast_mode.autocast("cuda", torch.float16):
            out: torch.Tensor = self._forward_org(
                x, x_mask, freqs_cis, adaln_input, transformer_options
            )
        if not torch.isfinite(out).all():
            if DEBUG_RECOMPUTE_PRINT:
                logger.info(
                    f"[{__tmp_module_name}] overflow detected, recomputing in fp32"
                )
            return self._forward_org(
                x, x_mask, freqs_cis, adaln_input, transformer_options
            )
        return out
    else:
        return self._forward_org(x, x_mask, freqs_cis, adaln_input, transformer_options)


def nextdit_forward_patch(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    context: torch.Tensor,
    num_tokens,
    attention_mask=None,
    transformer_options={},
    **kwargs,
):
    x.nan_to_num_()
    context.nan_to_num_()

    if x.dtype == torch.float16 and x.is_cuda:
        transformer_options = transformer_options.copy()
        transformer_options["__fp16_fallback_mode"] = None
        with torch.amp.autocast_mode.autocast("cuda", torch.float32):
            return self._forward_org(
                x,
                timesteps,
                context,
                num_tokens,
                attention_mask,
                transformer_options,
                **kwargs,
            )

    return self._forward_org(
        x,
        timesteps,
        context,
        num_tokens,
        attention_mask,
        transformer_options,
        **kwargs,
    )


def nextdit_init_patch(self, *args, **kwargs):
    self.__init_org(*args, **kwargs)

    if REDUCE_WEIGHT:
        self.register_load_state_dict_pre_hook(weight_scale_hook)

    for name, sub in self.named_modules():
        setattr(sub, "__tmp_module_name", name)


if ENABLE_PATCH:
    if ENABLE_TORCH_COMPILE:
        logger.info("[lunima fp16 patch] patching core code to enable torch compile")
        lumina.JointTransformerBlock.forward = torch.compile(
            lumina.JointTransformerBlock.forward
        )

    logger.info("[lunima fp16 patch] patching core code to enable fp32 fallback")

    lumina.NextDiT.__init_org = lumina.NextDiT.__init__
    lumina.NextDiT.__init__ = nextdit_init_patch

    lumina.NextDiT._forward_org = lumina.NextDiT._forward
    lumina.NextDiT._forward = nextdit_forward_patch

    if hasattr(lumina, "clamp_fp16"):
        lumina.clamp_fp16 = lambda x: x  # disable fp16 clamp, we need nan/inf.

    lumina.JointTransformerBlock._forward_org = lumina.JointTransformerBlock.forward
    lumina.JointTransformerBlock.forward = forward_with_fp32_fallback

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_fp16_accumulation = True
