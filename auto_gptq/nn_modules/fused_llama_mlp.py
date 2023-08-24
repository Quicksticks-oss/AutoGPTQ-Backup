import math
from logging import getLogger

import torch
from transformers.models.llama.modeling_llama import LlamaMLP

from ._fused_base import FusedBaseMLPModule
from ..utils.import_utils import TRITON_AVAILABLE

logger = getLogger(__name__)

quant_fused_matmul_248_kernel = None


class FusedLlamaMLPForQuantizedModel(FusedBaseMLPModule):
    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
    ):
        super().__init__()

        self.infeatures = gate_proj.infeatures
        self.intermediate_size = gate_proj.outfeatures
        self.outfeatures = down_proj.outfeatures
        self.bits = gate_proj.bits
        self.maxq = gate_proj.maxq

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        return self.down_proj(self.triton_llama_mlp(x))

    def triton_llama_mlp(self, x):
        with torch.cuda.device(x.device):
            out_shape = x.shape[:-1] + (self.intermediate_size, )
            x = x.reshape(-1, x.shape[-1])
            M, K = x.shape
            N = self.intermediate_size
            c = torch.empty((M, N), device=x.device, dtype=torch.float16)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
            quant_fused_matmul_248_kernel[grid](
                x, c, self.gate_proj.qweight,
                self.gate_proj.scales, self.gate_proj.qzeros, self.gate_proj.g_idx,
                self.up_proj.qweight,
                self.up_proj.scales, self.up_proj.qzeros, self.up_proj.g_idx,
                M, N, K,
                self.bits, self.maxq,
                x.stride(0), x.stride(1),
                self.gate_proj.qweight.stride(0), self.gate_proj.qweight.stride(1),
                c.stride(0), c.stride(1),
                self.gate_proj.scales.stride(0), self.gate_proj.qzeros.stride(0)
            )
            c = c.reshape(out_shape)
            return c

    @classmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        if not use_triton:
            logger.warning(f"skip module injection for {cls.__name__} not support integrate without triton yet.")
            return
        elif not TRITON_AVAILABLE:
            logger.warning(f"skip module injection for triton is not installed.")
            return

        for name, m in model.named_modules():
            if not isinstance(m, LlamaMLP):
                continue

            mlp = cls(m.gate_proj, m.down_proj, m.up_proj)

            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name

            setattr(parent, child_name, mlp)

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.intermediate_size

            if (k, n) not in kn_values:
                kn_values[(k, n)] = m

        logger.info(f'Found {len(kn_values)} unique fused mlp KN values.')
        logger.info('Warming up autotune cache ...')
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2 ** m
                for (k, n), (modules) in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    modules.triton_llama_mlp(a)
        del kn_values


__all__ = ["FusedLlamaMLPForQuantizedModel"]
