'''
This is functional with the following packages:

diffusers==0.18.2

INTRUCTIONS:
In the file, /raid/s2198939/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/__init__.py,
    comment out:
        from .pipelines import (
        AudioPipelineOutput,
        # ConsistencyModelPipeline,
        # DanceDiffusionPipeline,
        ...
        )
In the file, /raid/s2198939/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/pipelines/pipeline_utils.py,
    comment out:
    # from transformers.utils import SAFE_WEIGHTS_NAME as TRANSFORMERS_SAFE_WEIGHTS_NAME

In the file, /raid/s2198939/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/pipelines/__init__.py,
    comment out:
    # from .consistency_models import ConsistencyModelPipeline
    # from .dance_diffusion import DanceDiffusionPipeline

In the file, /raid/s2198939/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/pipelines/consistency_models/__init__.py
    comment out:
    # from .pipeline_consistency_models import ConsistencyModelPipeline

transformers==4.48.2
accelerate==0.21.0
datasets=2.19.0
torch==2.5.1+cu124
torchvision==0.20.1+cu124

'''


import argparse
from tqdm import tqdm

import torch

from optim_utils import *
from io_utils import *

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import AutoModel, AutoTokenizer

def main(args):
    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the components for RadEdit pipeline

    # 1. UNet
    unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet")

    # 2. VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    # 3. Text encoder and tokenizer
    text_encoder = AutoModel.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-BioViL-T",
        model_max_length=128,
        trust_remote_code=True,
    )

    # 4. Scheduler
    scheduler = DDIMScheduler(
        beta_schedule="linear",
        clip_sample=False,
        prediction_type="epsilon",
        timestep_spacing="trailing",
        steps_offset=1,
    )

    # 5. Pipeline
    pipe = LocalStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    )

    pipe = pipe.to(device)

    # # dataset
    # set_random_seed(args.gen_seed)
    dataset, prompt_key = get_dataset(args.dataset, pipe=None, max_num_samples=args.max_num_samples, shard=args.shard)

    # args.end = min(args.end, len(dataset))
    args.end = len(dataset)

    # generation
    print("generation")

    all_metrics = ["uncond_noise_norm", "text_noise_norm"]
    all_tracks = []

    for i in tqdm(range(args.start, args.end)):

        if(i % 100 == 0):
            print("#################### i = ", i)
        seed = i + args.gen_seed

        prompt = dataset[i]
        print("Prompt: ", prompt)

        ### generation
        set_random_seed(seed)
        outputs, track_stats = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            track_noise_norm=True,
        )

        uncond_noise_norm, text_noise_norm = (
            track_stats["uncond_noise_norm"],
            track_stats["text_noise_norm"],
        )

        curr_line = {}
        for metric_i in all_metrics:
            values = locals()[metric_i]
            curr_line[f"{metric_i}"] = values

        curr_line["prompt"] = prompt

        all_tracks.append(curr_line)
        print("\n")

    os.makedirs("det_outputs_radedit", exist_ok=True)
    write_jsonlines(all_tracks, f"det_outputs_radedit/shard_{args.shard}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--dataset", default='mimic')
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=None, type=int)
    parser.add_argument("--image_length", default=224, type=int)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_id", default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images_per_prompt", default=4, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--max_num_samples", default=None, type=int)
    parser.add_argument("--shard", default=None, type=int)

    args = parser.parse_args()

    main(args)
