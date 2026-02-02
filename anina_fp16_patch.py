# A very basic ComfyUI plugin to patch anima, which is based cosmos pt2,
# to use fp16 "safely" on old GPUs.
# Only tested for anima.
# Written by reakaakasky (https://civitai.com/user/reakaakasky).

# To clarify, this plugin isn't uploaded to GitHub because it's a very "dirty" plugin.
# "dirty" means it hot-patches/monkey patches the comfyui core code.
# This approach is terrible from a programming perspective.
# But this is the simplest approach I can think of.

# How to use:
# Put this file in the ComfyUI "custom_nodes" dir.
# Use "ModelComputeDtype" node and set dtype to "fp16".
# To disable the patch, remove the file, or rename the ".py" suffix, to something
# like ".disable", whatever.

# Version: v1 (2/2/2026)
# Version: v1.1 (2/2/2026) optimazed for torch.compile

import torch
import logging


logger = logging.getLogger(__name__)
logger.info("[anima fp16 patch] patch loading")

NODE_CLASS_MAPPINGS = {}

NODE_DISPLAY_NAME_MAPPINGS = {}

import comfy.ldm.cosmos.predict2 as p2

ampf16 = torch.autocast("cuda", dtype=torch.float16)
ampf32 = torch.autocast("cuda", dtype=torch.float32)


def p2_Block_init_patch(self: p2.Block, *args, **kwargs):
    self.__init_org(*args, **kwargs)

    self.adaln_modulation_self_attn.forward = ampf16(self.adaln_modulation_self_attn.forward)
    self.adaln_modulation_cross_attn.forward = ampf16(self.adaln_modulation_cross_attn.forward)

    self.self_attn.forward = ampf16(self.self_attn.forward)
    self.cross_attn.forward = ampf16(self.cross_attn.forward)
    self.mlp.forward = ampf16(self.mlp.forward)

    self.forward = ampf32(self.forward)


p2.Block.__init_org = p2.Block.__init__
p2.Block.__init__ = p2_Block_init_patch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_accumulation = True

logger.info("[anima fp16 patch] patch loaded")
