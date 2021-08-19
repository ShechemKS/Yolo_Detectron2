import argparse
from collections import OrderedDict
from copy import deepcopy
import logging
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog

from yolo import add_yolo_config
from inference_net import Predictor, run_on_image


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/yolov5-Full-FixFPN-FixPoint-lsq-M4F8L8.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="outputs/QAT-4bit/model_final.pth",
        metavar="FILE",
        help="path to weights file",
    )
    parser.add_argument(
        "--policy",
        default="configs/policy_yolov5m-test.txt",
        metavar="FILE",
        help="path to the test policy file",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output model."
        "If not given, defaults to directory from which weights are loaded",
    )
    parser.add_argument(
        "--input",
        default="../../datasets/coco/images/val2017/000000000139.jpg",
        metavar="FILE",
        help="input image to initialize the clip values",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.QUANTIZATION.policy = args.policy
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    if not args.output:
        from pathlib import Path
        args.output = Path(cfg.MODEL.WEIGHTS).parent
    else:
        args.output = Path(args.output)
    cfg.freeze()
    return cfg


def quantize_param(params, bit):
    f_max = torch.max(params).item()
    f_min = torch.min(params).item()
    single_sided = f_min >= 0 or f_max <= 0
    scale = (f_max - f_min) / (2 ** bit)
    q_min = 0 if single_sided else -2 ** (bit - 1)
    zero_point = q_min - f_min / scale
    quant_params = torch.round(params / scale - zero_point)
    dtype = torch.uint8 if single_sided else torch.int8
    quant_params = quant_params.to(dtype)
    return quant_params, scale, zero_point


def quantize_weights(layer, weight):
    epsilon = 1e-4
    assert hasattr(layer, 'quant_weight')
    assert layer.quant_weight.method == 'dorefa'
    assert layer.quant_weight.tag == 'wt'
    assert 'lsq' in layer.quant_weight.args.keyword
    assert 'symmetry' not in layer.quant_weight.args.keyword
    clip_val = layer.quant_weight.clip_val
    level_num = layer.quant_weight.level_num.item()
    bit = layer.quant_weight.bit
    assert bit <= 8
    assert 2 ** bit == level_num, level_num
    quant_weight = layer.quant_weight(weight)
    quant_weight = quant_weight / clip_val
    quant_weight = (quant_weight + 1.0) / 2
    quant_weight = quant_weight * (level_num - 1)
    int_weight = torch.round(quant_weight)
    assert torch.max(torch.abs(quant_weight - int_weight)) < epsilon
    int_weight = int_weight.byte()
    return int_weight, bit


def get_model_layer(model, k):
    modules = k.split('.')
    param = modules.pop()
    for i in range(len(modules)):
        if modules[i].isdigit():
            modules[i] = f'[{modules[i]}]'
        if i == len(modules) - 1 or modules[i + 1].isdigit():
            continue
        else:
            modules[i] = f'{modules[i]}.'
    layer = eval(f'model.{"".join(modules)}')
    return layer, param


def quantize_model(model, args, cfg):
    bit = 32    # starting bit value.
    # Iterate over the state_dict of the model and create a new state dict
    # with quantized values
    include_params = ['level_num',
                      'clip_val',
                      'iteration',
                      'num_batches_tracked',
                      'anchors',
                      'anchor_grid']
    quant_dict = OrderedDict()
    fp_dict = deepcopy(model.state_dict())

    for k, v in fp_dict.items():
        logger.info(f'processing {k}')
        if any(x in k for x in include_params):
            quant_dict[k] = v
            continue
        layer, param = get_model_layer(model, k)
        # Quantization layer
        logger.info(f'quantizing {k}')
        if hasattr(layer, 'quantization'):
            if param == 'weight':
                quantized_weight, bit = quantize_weights(layer, v)
                quant_dict[k] = quantized_weight
            elif param == 'bias':
                quant_param, scale, zero_point = quantize_param(v, bit)
                quant_dict[k] = quant_param
                quant_dict[f'{k}.{param}.scale'] = scale
                quant_dict[f'{k}.{param}.zero_point'] = zero_point
            else:
                raise RuntimeError(param)
        elif hasattr(layer, 'tag') and layer.tag == 'norm':
            bias, scale = l.quantize()
            quant_dict[k] = quant_param
            quant_dict[f'{k}.{param}.scale'] = scale
            quant_dict[f'{k}.{param}.zero_point'] = zero_point
        else:
            raise RuntimeError(layer)
    logger.info('quantization complete')
    return quant_dict


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )
    predictor = Predictor(cfg)
    img = read_image(args.input, format="BGR")
    # Need to run a dummy input to initialize the clip values
    predictions, visualized_output = run_on_image(img, predictor, metadata)
    model = predictor.model

    quant_dict = quantize_model(model, args, cfg)
    if args.output.is_dir():
        args.output = args.output / 'quantized_model.pth'
    torch.save(quant_dict, args.output)
    logger.info(f'Quantized state dictionary saved to {args.output}')
