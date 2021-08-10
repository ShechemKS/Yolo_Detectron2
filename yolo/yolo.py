import logging
import math
import numpy as np
import torch
from torch import nn, Tensor
from typing import List, Dict, Tuple

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling import build_backbone

from .general import non_max_suppression, scale_coords
from .loss import ComputeLoss


__all__ = ["Yolo"]

logger = logging.getLogger(__name__)


def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N,Ai,H,W,K) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 5, tensor.shape
    N = tensor.shape[0]
    tensor = tensor.view(N, -1, K)  # Size=(N,HWA,K)
    return tensor


@META_ARCH_REGISTRY.register()
class Yolo(nn.Module):
    """
    Implement Yolo
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        loss,
        num_classes,
        conf_thres,
        iou_thres,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head

        self.num_classes = num_classes
        self.single_cls = num_classes == 1
        # Inference Parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss = loss
        # self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        # self.loss_normalizer_momentum = 0.9
        self.init_stride()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = list(backbone_shape.values())
        head = YoloHead(cfg, feature_shapes)
        loss = ComputeLoss(cfg, head)
        return{
            "backbone": backbone,
            "head": head,
            "loss": loss,
            "num_classes": head.nc,
            "conf_thres": cfg.MODEL.YOLO.CONF_THRESH,
            "iou_thres": cfg.MODEL.YOLO.IOU_THRES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.

        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(
            results
        ), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def init_stride(self):
        s = 256  # 2x min stride
        dummy_input = torch.zeros(1, len(self.pixel_mean), s, s)
        features = self.backbone(dummy_input)
        features = list(features.values())
        pred = self.head(features)
        self.head.stride = torch.tensor(
            [s / x.shape[-2]
                for x in pred])  # forward
        self.head.anchors /= self.head.stride.view(-1, 1, 1)
        self.stride = self.head.stride
        self.head._initialize_biases()  # only run once
        self.loss._initialize_ssi(self.stride)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = list(features.values())

        pred = self.head(features)
        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.loss(pred, gt_instances)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        pred, images.image_sizes
                    )
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(pred, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def inference(self, x, image_sizes):
        """
        Returns:
        z (Tensor) : [N, nl*na*(sum of grid sizes) , no] indictaing
                    1. Box position z[..., 0:2]
                    2. Box width and height z[..., 2:4]
                    3. Objectness z[..., 5]
                    4. Class probabilities z[..., 6:]
        """
        z = []
        for i in range(self.head.nl):
            # x(bs,na,ny,nx,no)
            bs, _, ny, nx, _ = x[i].shape
            if self.head.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.head.grid[i] = self.head._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            # if self.head.inplace:
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.head.grid[i]) * self.head.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.head.anchor_grid[i]  # wh
            # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            #     xy = (y[..., 0:2] * 2. - 0.5 + self.head.grid[i]) * self.head.stride[i]  # xy
            #     wh = (y[..., 2:4] * 2) ** 2 * self.head.anchor_grid[i].view(1, self.head.na, 1, 1, 2)  # wh
            #     y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, self.head.no))
        return self.process_inference(torch.cat(z, 1), image_sizes)

    def process_inference(self, out, image_sizes):
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True, agnostic=self.single_cls)
        assert len(out) == len(image_sizes)
        results_all: List[Instances] = []
        # Statistics per image
        for si, (pred, img_size) in enumerate(zip(out, image_sizes)):

            if len(pred) == 0:
                continue

            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            # Predn shape [ndets, 6] of format [xyxy, conf, cls] relative to the input image size
            result = Instances(img_size)
            result.pred_boxes = Boxes(predn[:, :4])  # TODO: Check if resizing needed
            result.scores = predn[:, 4]
            result.pred_classes = predn[:, 5]   # TODO: Check the classes
            results_all.append(result)
        return results_all

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class YoloHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        nc,
        anchors,
    ):

        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        assert self.nl == len(input_shape)
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(
            self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        ch = [x.channels for x in input_shape]
        self.m = nn.ModuleList(Conv2d(x, self.no * self.na, 1) for x in ch)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        model_yaml_file = cfg.MODEL.YAML
        import yaml  # for torch hub
        with open(model_yaml_file) as f:
            model_yaml = yaml.safe_load(f)  # model dict
        anchors = model_yaml['anchors']
        nc = model_yaml['nc']
        return {
            "input_shape": input_shape,
            "nc": nc,
            "anchors": anchors,
        }

    def forward(self, x: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            x (list[Tensor]): #nl tensors,
                                each having shape [N, na, Hi, Wi, nc + 5]
            z (Tensor) : [N, nl*na*(sum of grid sizes) , no] indictaing
                    1. Box position z[..., 0:2]
                    2. Box width and height z[..., 2:4]
                    3. Objectness z[..., 5]
                    4. Class probabilities z[..., 6:]
        """
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99)
                                      ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
