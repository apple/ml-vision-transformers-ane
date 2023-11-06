import pytest
from vision_transformers.model import _build_model, get_stage_strides
import torch
from export import moat_export
import os


@pytest.mark.parametrize("image_shape", [(512, 512), (256, 256)])
@pytest.mark.parametrize("output_stride", [8, 16, 32])
@pytest.mark.parametrize("attention_mode", ["global", "local"])
@pytest.mark.parametrize(
    "base_arch",
    [
        "tiny-moat-0",
        "tiny-moat-1",
        "tiny-moat-2",
    ],
)
def test_model(output_stride, base_arch, image_shape, attention_mode):
    """MOAT unit test

    :param output_stride: param base_arch:
    :param image_shape: param attention_mode:
    :param base_arch:
    :param attention_mode:

    """
    image_height, image_width = image_shape
    with torch.no_grad():
        model_config, model = _build_model(
            shape=(1, 3, image_height, image_width),
            base_arch=base_arch,
            attention_mode=attention_mode,
            output_stride=output_stride,
        )
        stage_stride = get_stage_strides(output_stride)
        inputs = torch.zeros((1, 3, image_height, image_width), device="cpu")
        output = model(inputs)
        assert len(output) == 4
        output_h, output_w = image_height // 2, image_width // 2
        for stage_idx, stride in enumerate(stage_stride):
            assert len(output[stage_idx].shape) == 4
            output_h, output_w = output_h // stride, output_w // stride
            assert output_h == output[stage_idx].shape[-2]
            assert output_w == output[stage_idx].shape[-1]


def test_export():
    """ """
    moat_export(
        base_arch="tiny-moat-0",
        shape=(1, 3, 256, 256),
        attention_mode="local",
    )
    assert os.path.exists(
        "exported_model/tiny-moat-0_localAttn_batch1_256x256_PEType.LePE_ADD_split-head_True.mlpackage"
    )
