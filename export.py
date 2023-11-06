#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#
import torch
import coremltools as ct

from vision_transformers.attention_utils import (
    PEType,
)
from vision_transformers.model import _build_model


def moat_export(
    base_arch="tiny-moat-0",
    shape=(1, 3, 256, 256),
    pe_type=PEType.LePE_ADD,
    attention_mode="local",
):
    """

    :param base_arch:  (Default value = "tiny-moat-0")
    :param shape:  (Default value = (1)
    :param 3:
    :param 256:
    :param 256):
    :param pe_type:  (Default value = PEType.LePE_ADD)
    :param attention_mode:  (Default value = "local")

    """
    split_head = True
    batch = shape[0]
    print("****** batch_size: ", batch)
    pe_type = pe_type if "moat" in base_arch else "ape"
    attention_mode = attention_mode if "moat" in base_arch else "global"
    local_window_size = [8, 8] if attention_mode == "local" else None
    print("****** building model: ", base_arch)
    if "tiny-moat" in base_arch:
        _, model = _build_model(
            base_arch=base_arch,
            shape=shape,
            split_head=split_head,
            pe_type=pe_type,
            channel_buffer_align=False,
            attention_mode=attention_mode,
            local_window_size=local_window_size,
        )
    resolution = f"{shape[-2]}x{shape[-1]}"

    x = torch.rand(shape)

    with torch.no_grad():
        model.eval()
        traced_optimized_model = torch.jit.trace(model, (x,))
        ane_mlpackage_obj = ct.convert(
            traced_optimized_model,
            convert_to="mlprogram",
            inputs=[
                ct.TensorType("x", shape=x.shape),
            ],
        )

        out_name = f"{base_arch}_{attention_mode}Attn_batch{batch}_{resolution}_{pe_type}_split-head_{split_head}"
        out_path = f"./exported_model/{out_name}.mlpackage"
        ane_mlpackage_obj.save(out_path)

        import shutil

        shutil.make_archive(f"{out_path}", "zip", out_path)


if __name__ == "__main__":
    base_arch = "tiny-moat-0"
    attention_mode = ["global", "local"]
    pe_type = PEType.SINGLE_HEAD_RPE
    shapes = [[1, 3, 512, 512], [1, 3, 256, 256]]
    bs = [1]
    for att_mode in attention_mode:
        for shape in shapes:
            for batch in bs:
                shape[0] = batch
                moat_export(
                    base_arch,
                    shape,
                    pe_type=pe_type,
                    attention_mode=att_mode,
                )
