# Point Cloud Registration in Jittor

Jittor implementation of these papers:

[Geometric Transformer for Fast and Robust Point Cloud Registration](https://arxiv.org/abs/2202.06688).

[Deep Graph-Based Spatial Consistency for Robust Non-Rigid Point Cloud Registration](http://arxiv.org/abs/2303.09950).

[Learning Instance-Aware Correspondences for Robust Multi-Instance Point Cloud Registration in Cluttered Scenes](https://arxiv.org/abs/2404.04557).


## News

2024.09.24: Codes and pretrained models release.

## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n pcr-jittor python==3.8
conda activate pcr-jittor

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop

2. Install vision3d following https://github.com/qinzheng93/vision3d
```

Code has been tested with Ubuntu 20.04, GCC 9.3.0, Python 3.8, PyTorch 1.7.1, CUDA 11.1 and cuDNN 8.1.0.

## Pre-trained Weights

We provide pretrained weights in https://drive.google.com/drive/folders/1zb6TzkUtM8uGy7qBprGhg1oufIjxxZDZ?usp=drive_link.

### Training

 Use the following command for training.

```bash
./train_3dmatch_jittor.sh
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/3dmatch/test.py
```


## Citation

```bibtex
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}

@misc{qin2022geometric,
      title={Geometric Transformer for Fast and Robust Point Cloud Registration},
      author={Zheng Qin and Hao Yu and Changjian Wang and Yulan Guo and Yuxing Peng and Kai Xu},
      year={2022},
      eprint={2202.06688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{qin2023deep,
    title={Deep Graph-Based Spatial Consistency for Robust Non-Rigid Point Cloud Registration},
    author={Zheng Qin and Hao Yu and Changjian Wang and Yuxing Peng and Kai Xu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2023},
    pages={5394-5403}
}

@inproceedings{yu2024learning,
  title={Learning Instance-Aware Correspondences for Robust Multi-Instance Point Cloud Registration in Cluttered Scenes},
  author={Yu, Zhiyuan and Qin, Zheng and Zheng, Lintao and Xu, Kai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19605--19614},
  year={2024}
}
```

## Acknowledgements

- [vision3d](https://github.com/qinzheng93/vision3d)
- [Geotransformer](https://github.com/qinzheng93/GeoTransformer)
- [GraphSCNet](https://github.com/qinzheng93/GraphSCNet)
