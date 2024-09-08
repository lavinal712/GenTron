import argparse
import logging
import numpy as np
import os
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import OrderedDict
from copy import deepcopy
from diffusers.models import AutoencoderKL
from glob import glob
from PIL import Image
from time import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, CLIPTextModel
from tqdm import tqdm

from diffusion import create_diffusion
from models import GenTron_models


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features_path", type=str, default="features")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(GenTron_models.keys()), default="GenTron-T2I-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args=None):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0:
        os.makedirs(args.features_path, exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_features'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_captions'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, 'imagenet256_masks'), exist_ok=True)
    
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    text_config = AutoConfig.from_pretrained(args.text_encoder)
    embed_dim = text_config.projection_dim
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    requires_grad(vae, False)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device)
    requires_grad(text_encoder, False)
    with open("data/imagenet1000_clsidx_to_labels.txt", "r") as f:
        id2label = eval(f.read())

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    index = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        y_int = y.item()
        labels = [label.strip() for label in id2label[y.item()].split(",")]
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            for i, label in enumerate(labels):
                if not os.path.exists(f"{args.features_path}/imagenet256_captions/{y_int}-{i}.npy"):
                    y = [label]
                    y_inputs = tokenizer(y, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
                    tokens = y_inputs["input_ids"].to(device)
                    y = text_encoder(input_ids=tokens).last_hidden_state
                    mask = y_inputs["attention_mask"].bool()
                
                    y = y.detach().cpu().numpy()
                    np.save(f"{args.features_path}/imagenet256_captions/{y_int}-{i}.npy", y)
                    mask = mask.detach().cpu().numpy()
                    np.save(f"{args.features_path}/imagenet256_masks/{y_int}-{i}.npy", mask)
            
            if not os.path.exists(f"{args.features_path}/imagenet256_features/{y_int}-{index}.npy"):
                index = 0
            else:
                index += 1
            x = x.detach().cpu().numpy()
            np.save(f"{args.features_path}/imagenet256_features/{y_int}-{index}.npy", x)


if __name__ == "__main__":
    args = parse_args()
    main(args)
