# Copyright (c) Facebook, Inc. and its affiliates.
from pathlib import Path
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec
from .common import Conv, C3, SPP, Concat, Focus
from .general import make_divisible


class DarkNet(Backbone):

    def __init__(self, cfg='yolov5.yaml', ch=3):  # model, input channels, number of classes
        super().__init__()
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        ch = [ch]
        c2 = ch[-1]
        layers = []
        save = []
        head = [x for x in self.yaml['head'] if 'Detect' not in x]
        self.out_features = -1
        self.out_feature_channels = {}
        self.out_feature_strides = {}
        if 'Detect' in self.yaml['head'][-1]:
            self.out_features = self.yaml['head'][-1][0]
            print("Detection Head found")
        # Define model
        gd, gw = self.yaml['depth_multiple'], self.yaml['width_multiple']
        for i, (f, n, m, args) in enumerate(self.yaml['backbone'] + head):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a
                except BaseException:
                    pass
            n = max(round(n * gd), 1) if n > 1 else n
            if m in [Conv, C3, SPP, Focus]:
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m is C3:
                    args.insert(2, n)
                    n = 1
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            else:
                c2 = ch[f]
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in m_.parameters()])  # number params
            m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            layers.append(m_)
            save.extend(
                x %
                i for x in (
                    [f] if isinstance(
                        f,
                        int) else f) if x != -
                1)  # append to savelist
            if i == 0:
                ch = []
            ch.append(c2)
            if i in self.out_features:
                self.out_feature_channels[i] = c2
                self.out_feature_strides[i] = 1
        save.extend(x for x in self.out_features if self.out_features != -1)
        self.model = nn.Sequential(*layers)
        print(self.model)
        self.save = sorted(save)
        print(self.save)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', False)

        # Init weights, biases
        self.initialize_weights()

    def forward(self, x):
        return self.forward_augment(x)  # augmented inference, None

    def forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:   # Not the previous layer
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)  # run
            y.append(x if m.i in self.save else None)

        return {j: y[j] for j in self.out_features}

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            print(t)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.out_feature_channels[name],
                stride=self.out_feature_strides[name]
            )
            for name in self.out_features
        }


@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg, input_shape):
    model_yaml_file = cfg.MODEL.YAML
    import yaml  # for torch hub
    with open(model_yaml_file) as f:
        model_yaml = yaml.safe_load(f)  # model dict
    in_channels = 3
    return DarkNet(model_yaml, in_channels)
