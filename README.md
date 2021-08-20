# Yolo for Detectron2
Implementation of Yolo using [Facebook's Detectron2 Framework](https://github.com/facebookresearch/detectron2).

With added quantization support following the work in
> Chen, Peng, et al. "Aqd: Towards accurate quantized object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

## Description

This repo implements YoloV5 within Facebook's Detectron2 framework. 
Currently, only YoloV5m has been fully tested. 
Support is included for YoloV4-tiny. 
Support will be extended for other Yolo versions.

This repo also enables quantization and quantization-aware-training using the framework provided in 
> Chen, Peng, et al. "Aqd: Towards accurate quantized object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

The quantization framework is implemented in 
[QTool: A low-bit quantization toolbox for deep neural networks in computer vision](https://github.com/MonashAI/QTool).
Use the quantization branch to train and test quantized models.

## Setup
- This repo requires installation of the Detectron2 framework. 
- Installation instructions can be found on their own [GitHub page](https://github.com/facebookresearch/detectron2).
- However, if you intend on using the quantization modules, 
follow the instructions given in [QTool - Detection](https://github.com/MonashAI/QTool/blob/master/doc/detectron2.md) 
to install the quantization fork of the detectron framework. Currently, Yolo has only been tested within this framework. 
- Once Detectron2 has been installed, clone this repo. You can clone it anywhere you like, but it is convenient to clone it to the `detectron2/projects` directory for consistency.
```
git clone https://github.com/ShechemKS/Yolo_Detectron2.git
```
That's it! It will just work. 

## Training and Inference

To train the model run
```
python train_net.py --config-file configs/yolov5-Full.yaml
```

You may include any of the usual Detectron2 config options. 

To use the model for inference, run
```
python inference_net.py --config-file configs/yolov5-Full.yaml --inputs path/to/image-dir/ --output path/to/save-dir/

```

## TODO
- [x] Create a non-quantization branch for compatibility with the main detectron framework
- [ ] Add support for other Yolo versions

## References
- [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- [MonashAI/QTool](https://github.com/MonashAI/QTool)
