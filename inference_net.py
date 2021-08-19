import os
import argparse
import torch
import tqdm
import time

from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.transforms import Resize
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from yolo import add_yolo_config


class Predictor(DefaultPredictor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.aug = Resize(cfg.INPUT.SIZE)


def run_on_image(image, predictor, metadata):
    """
    Args:
        image (np.ndarray): an image of shape (H, W, C) (in BGR order).
            This is the format used by OpenCV.

    Returns:
        predictions (dict): the output of the model.
        vis_output (VisImage): the visualized image output.
    """
    cpu_device = torch.device('cpu')
    predictions = predictor(image)
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    image = image[:, :, ::-1]
    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

    instances = predictions["instances"].to(cpu_device)
    vis_output = visualizer.draw_instance_predictions(predictions=instances)

    return predictions, vis_output


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
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
    cfg.freeze()
    return cfg


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
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = run_on_image(img, predictor, metadata)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
                visualized_output.save(out_filename)
