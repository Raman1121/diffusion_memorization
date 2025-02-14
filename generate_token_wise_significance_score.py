import torch
import torchvision
import torchvision.transforms as transformsdd
import mediapy as media
import numpy as np
from diffusers import DDIMScheduler

try:
    from local_sd_pipeline import LocalStableDiffusionPipeline
    from optim_utils import *
except ModuleNotFoundError:
    import os; os.chdir("..")
    from local_sd_pipeline import LocalStableDiffusionPipeline
    from optim_utils import *

from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pickle

def main(args):

    ## Load the RadEdit pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    num_inference_steps = 50
    guidance_scale = 7.5
    num_images_per_prompt = 4
    image_size = 512

    ## Load the dataframe containing the memorized prompts
    df = pd.read_csv(args.memorized_prompts_path)
    all_prompts = df["prompt"].tolist()

    # all_prompts = all_prompts[:1]

    # Divide the dataframe into args.num_shards shards
    if(args.num_shards is not None):
        print("Dividing the dataframe into {} shards".format(args.num_shards))
        print("Selected shard: ", args.shard)
        all_prompts = np.array_split(all_prompts, args.num_shards)[args.shard]

    # Store the token-wise significance score for each prompt
    TOKEN_GRAD_NORMS = {}

    for prompt in tqdm(all_prompts):
        print("Prompt: ", prompt)
        token_grads = pipe.get_text_cond_grad(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            target_steps=list(range(num_inference_steps)),
        )
        torch.cuda.empty_cache()

        prompt_tokens = pipe.tokenizer.encode(prompt)
        prompt_tokens = prompt_tokens[1:-1]
        prompt_tokens = prompt_tokens[:tokenizer.model_max_length]
        token_grads = token_grads[1:(1+len(prompt_tokens))]
        token_grads = token_grads.cpu().tolist()

        all_tokes = []

        for curr_token in prompt_tokens:
            all_tokes.append(pipe.tokenizer.decode(curr_token))
            
        # Append the token and the corresponding gradient to the dictionary
        # Key: token, Value: a list containing the gradient for that token
        for token, grad in zip(all_tokes, token_grads):
            if token not in TOKEN_GRAD_NORMS:
                TOKEN_GRAD_NORMS[token] = []
            TOKEN_GRAD_NORMS[token].append(grad)

    # Save this dictionary to a dataframe
    # import pdb; pdb.set_trace()

    # try:
    #     df_token_grad_norm = pd.DataFrame.from_dict(TOKEN_GRAD_NORMS, orient='index')
        
    #     # Rename the first column to 'token'
    #     df_token_grad_norm = df_token_grad_norm.reset_index()
    #     df_token_grad_norm = df_token_grad_norm.rename(columns={'index': 'token'})

    #     # Fill nan values with 0
    #     # df_token_grad_norm = df_token_grad_norm.fillna(0)

    #     # Create a new column 'avg_scores' which contains the average of the scores for each token
    #     df_token_grad_norm['avg_scores'] = df_token_grad_norm.iloc[:, 1:].mean(axis=1)

    #     # Sort the dataframe based on the 'avg_scores' column
    #     df_token_grad_norm = df_token_grad_norm.sort_values(by='avg_scores', ascending=False).reset_index(drop=False)

    #     # Drop the column 'index'
    #     df_token_grad_norm = df_token_grad_norm.drop(columns=['index'])

    #     df_token_grad_norm.to_csv(os.path.join(args.output_path, 'token_significance_scores.csv'), index=False)
    # except:
    #     import pdb; pdb.set_trace()

    # Save the dictionary to a pickle file
    try:
        with open(os.path.join(args.output_path, 'token_significance_scores.pkl'), 'wb') as f:
            pickle.dump(TOKEN_GRAD_NORMS, f)
    except:
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--memorized_prompts_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default='/raid/s2198939/diffusion_memorization/det_outputs_radedit', required=False)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard", type=int, default=None)
    args = parser.parse_args()

    main(args)

