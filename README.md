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
|                                             |Backbone|   AP    |   AP@50   |   AP@75   |   AP_S    |   AP_M    |    AP_L   | download | 
|---------------------------------------------|:-------:|:-------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| MTL-Det |[ResNeXt-101-64×4](work_dirs/HTL_1x_renext/HTL_cascade_rcnn_x101_64x4d_fpn_1x_hrsid.py)| 68.0 | 89.5 |  77.7 | 68.7 | 69.6 |25.8 |[Google](https://drive.google.com/file/d/1I1OZ4Aqu7XF_6olL9E0MCEkAMgrLJaGl/view?usp=sharing)

Put the model to ***work_dirs/HTL_1x_renext/***
#### LSSDD-v1.0 dataset
|                                             |Backbone|Off-shore|In-shore |  ALL  | download | 
|---------------------------------------------|:-------:|:-------:|:---------:|:---------:|:---------:|
| MTL-Det |[ResNet-50](work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py)| 88.7 | 38.7 |  71.7 |[Google](https://drive.google.com/file/d/1kzTY-dijPJQM2GWmw0erzrCDdsOSxr28/view?usp=sharing)

Put the model to ***work_dirs/HTL_1x_faster/***


## Evaluate
1.Multi-GPUs Test
```shell
./tools/dist_test.sh work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py work_dirs/HTL_1x_faster/epoch_11.pth 8 --eval mAP
```
2.Single-GPU Test
```shell
python tools/test.py work_dirs/HTL_1x_faster/HTL_ins_faster_rcnn_r50_fpn_1x_hrsid.py work_dirs/HTL_1x_faster/epoch_11.pth --eval mAP
```

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
