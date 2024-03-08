from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime
import onnxsim
import segmentation_models_pytorch as smp
import torch

from src.logger import LOGGER

ONNX_PROVIDERS = (
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
)


def convert_onnx(  # noqa: WPS210
    model: torch.nn.Module,
    out_path: str,
    input_h: int,
    input_w: int,
    logit_tolerance: float = 5e-5,
) -> None:
    model.eval()
    dumpy_input = torch.randn((1, 3, input_h, input_w), dtype=torch.float32, device='cpu')

    with torch.no_grad():
        ref_logits = model(dumpy_input)

    LOGGER.info('=> model convering to onnx')
    torch.onnx.export(
        model,
        dumpy_input,
        out_path,
        export_params=True,
        verbose=True,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
    )
    LOGGER.info('=> model has been convered to onnx')
    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=False,
        overwrite_input_shapes={'input': [1, 3, input_h, input_w]},
    )
    onnx.save(model_onnx, out_path)
    LOGGER.info('simply onnx model')

    ort_session = onnxruntime.InferenceSession(
        out_path,
        providers=ONNX_PROVIDERS,
    )

    onnx_input_arr = dumpy_input.numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input_arr}
    ort_outputs = ort_session.run(None, ort_inputs)

    logits = ort_outputs[0]
    if not np.allclose(logits, ref_logits.numpy(), atol=logit_tolerance):
        raise ValueError(f'Logits did not match with tolerance threshold {logit_tolerance}')


def lightning_to_regular_state_dict(lightning_state_dict: Dict[str, torch.Tensor | Any]) -> Dict[str, torch.Tensor]:
    regular_state_dict = {}
    for key, state_val in lightning_state_dict.items():
        # Filter out keys specific to PyTorch Lightning
        if key.startswith('model.'):
            regular_key = key[len('model.') :]
            regular_state_dict[regular_key] = state_val
    return regular_state_dict


def main() -> None:
    # TODO: refactor it into a callback, instead of a script with hard-coded values.

    model = smp.create_model(arch='FPN', encoder_name='efficientnet-b0', classes=20)
    input_path = (
        '/home/egor/Projects/human_body_segmentation/src/lightning_logs/version_2/checkpoints/epoch=29-step=8760.ckpt'
    )
    outonnx_path = 'model.onnx'
    LOGGER.info('=> loading model from {0}'.format(input_path))

    lightning_ckpt = torch.load(input_path, map_location='cpu')
    state_dict = lightning_to_regular_state_dict(lightning_ckpt['state_dict'])
    model.load_state_dict(state_dict)
    model.eval()
    convert_onnx(model, outonnx_path, input_h=224, input_w=224)


if __name__ == '__main__':
    main()
