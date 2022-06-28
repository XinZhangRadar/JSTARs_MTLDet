# Multitask learning for ship detection from synthetic aperture radar images
This is the official implementation of ***MTL-Det*** (JSTARs), a SAR ship detection method. For more details, please refer to:

**Multitask learning for ship detection from synthetic aperture radar images [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9508842)**  <br />
Xin Zhang , Chunlei Huo , Nuo Xu , Hangzhi Jiang, Yong Cao, Lei Ni and Chunhong Pan<br />

## Installation

Please refer to [install.md](docs/install.md) for installation and dataset preparation.


## Getting Started
## Preparation
Clone the code
```
git clone https://github.com/XinZhangNLPR/JSTARs_MTLDet.git
```


Download the model weight used in the paper:

#### HRSID dataset
|                                             | AP | AP@50  |AP@75 | 
|---------------------------------------------|-------:|:-------:|:---------:|
| [ResNeXt](OpenPCDet/tools/cfgs/kitti_models/pv_rcnn_focal_lidar.yaml) | 83.91 | 85.20 | [Google](https://drive.google.com/file/d/1XOpIzHKtkEj9BNrQR6VYADO_T5yaOiJq/view?usp=sharing) \| [Baidu](https://pan.baidu.com/s/1t1Gk8bDv8Q_Dd5vB4VtChA) (key: m15b) |
| [PV-RCNN + Focals Conv (multimodal)](OpenPCDet/tools/cfgs/kitti_models/pv_rcnn_focal_multimodal.yaml) | 84.58 | 85.34 | [Google](https://drive.google.com/file/d/183araPcEmYSlruife2nszKeJv1KH2spg/view?usp=sharing) \| [Baidu](https://pan.baidu.com/s/10XodrSazMFDFnTRdKIfbKA) (key: ie6n) |
| [Voxel R-CNN (Car) + Focals Conv (multimodal)](OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml) | 85.68 | 86.00 | [Google](https://drive.google.com/file/d/1M7IUosz4q4qHKEZeRLIIBQ6Wj1-0Wjdg/view?usp=sharing) \| [Baidu](https://pan.baidu.com/s/1bIN3zDmPXrURMOPg7pukzA) (key: tnw9) |
* [ResNet101](https://drive.google.com/file/d/194cUFKymSdYq8GNQR4ZgzPUse-pG8ch9/view?usp=sharing)
## Citation

```
@article{zhang2021multitask,
  title={Multitask learning for ship detection from synthetic aperture radar images},
  author={Zhang, Xin and Huo, Chunlei and Xu, Nuo and Jiang, Hangzhi and Cao, Yong and Ni, Lei and Pan, Chunhong},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={14},
  pages={8048--8062},
  year={2021},
  publisher={IEEE}
}
```
