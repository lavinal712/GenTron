## GenTron: Diffusion Transformers for Image and Video Generation

### Unofficial PyTorch Implementation

### [Paper](https://arxiv.org/abs/2312.04557) | [Project Page](https://www.shoufachen.com/gentron_website)

> [**GenTron: Diffusion Transformers for Image and Video Generation**](https://www.shoufachen.com/gentron_website)</br>
> Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, Juan-Manuel Perez-Rua
> <br>The University of Hong Kong, Meta</br>

This repository contains:

* 🪐 A simple PyTorch [implementation](models.py) of Text-to-Image GenTron
* 🛸 A GenTron [training script](train.py)

## Setup

[DiT](https://github.com/facebookresearch/DiT) and [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha)

## Sampling

```bash
python sample.py --image_size 512 --seed 1
```

```bash
python sample.py --model GenTron-T2I-XL/2 --image_size 256 --ckpt /path/to/model.pt
```

## Training

### Training GenTron

```bash
accelerate launch --mixed_precision fp16 train.py --data_path /path/to/ImageNet/train
```

## Acknowledgments

- [DiT](https://github.com/facebookresearch/DiT)
- [fast-DiT](https://github.com/chuanyangjin/fast-DiT)
- [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha)
