import argparse
import torch
from diffusers.models import AutoencoderKL
from download import find_model
from einops import rearrange, repeat
from torchvision.io import write_video
from torchvision.utils import save_image
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, CLIPTextModel

from diffusion import create_diffusion
from models import GenTron_models


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=list(GenTron_models.keys()), default="GenTron-T2V-XL/2")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--mfg_scale", type=float, default=1.3)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a GenTron checkpoint.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "GenTron-T2V-XL/2", "Only GenTron-T2V-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    latent_size = args.image_size // 8
    text_config = AutoConfig.from_pretrained(args.text_encoder)
    embed_dim = text_config.d_model if args.model == "GenTron-T2V-G/2" else text_config.projection_dim
    model = GenTron_models[args.model](
        input_size=latent_size,
        embedding_dim=embed_dim,
    ).to(device)
    ckpt_path = args.ckpt or f"GenTron-T2V-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device)

    z = torch.randn(3, 4, args.video_length, latent_size, latent_size, device=device)
    y = ["a few cartoon characters are having a conversation", "a few cartoon characters are having a conversation", "a few cartoon characters are having a conversation"]
    y_inputs = tokenizer(y, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    y_null_inputs = tokenizer([""] * 3, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    tokens = y_inputs["input_ids"].to(device)
    uncond_tokens = y_null_inputs["input_ids"].to(device)
    y = text_encoder(input_ids=tokens).last_hidden_state
    y_null = text_encoder(input_ids=uncond_tokens).last_hidden_state
    mask = y_inputs["attention_mask"].bool().to(device)
    uncond_mask = y_null_inputs["attention_mask"].bool().to(device)

    z = torch.cat([z, z], 0)
    y = torch.cat([y, y_null], 0)
    mask = torch.cat([mask, uncond_mask], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, mfg_scale=args.mfg_scale, mask=mask)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg_and_mfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    b, _, _, _, _ = samples.shape
    samples = rearrange(samples, "b c f h w -> (b f) c h w").contiguous()
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, "(b f) c h w -> b c f h w", b=b).contiguous()

    for i, sample in enumerate(samples):
        sample = sample.squeeze(0)
        sample = rearrange(sample, "c f h w -> f h w c").contiguous()
        write_video(f"{i}.mp4", sample.cpu(), args.fps, "h264")


if __name__ == "__main__":
    args = parse_args()
    main(args)
