import os
import csv
import numpy as np
import argparse
import random
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dataset import SiameseNetworkDataset
from model import SiameseNetwork

from sklearn import metrics
import Utils

from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline, AutoencoderKL
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

def get_reidentification_score(model, img1, img2, transforms):
    with torch.no_grad():
        img1 = transforms(img1).unsqueeze(0).to('cuda')
        img2 = transforms(img2).unsqueeze(0).to('cuda')

        try:
            outputs = model(img1, img2)
        except:
            import pdb; pdb.set_trace()
        score = torch.sigmoid(outputs)

    return score.item()

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate ReID scores between memorized and non-memorized prompts.")
    parser.add_argument("--run_on", type=str, default="mem_prompts", help="Choose between 'mem_prompts' and 'non_mem_prompts'.")
    parser.add_argument("--num_shards", type=int, default=0, help="Number of shards.")
    parser.add_argument("--shard", type=int, help="Shard index")
    return parser.parse_args()

####################################################################################################

"""
We need to calculate the re-id score between:
1. The ground truth image corresponding to the prompt
2. Multiple generated images with the same prompt
"""

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# RadEdit Model: https://huggingface.co/microsoft/radedit
def load_sd_pipeline():
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

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None
    )

    pipe = pipe.to(device)

    return pipe

def generate_synthetic_images(pipe, prompt, num_images_per_prompt=1, seed=42):

    NUM_INF_STEPS = 100
    GUIDANCE_SCALE = 7.5
    NUM_IMAGES_PER_PROMPT = num_images_per_prompt

    generator = torch.Generator("cuda").manual_seed(seed)

    image = pipe(
        prompt,
        generator=generator,
        num_inference_steps=NUM_INF_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]

    return image

# RadDino Model: https://huggingface.co/microsoft/rad-dino
def load_image_encoder():
    repo = "microsoft/rad-dino"
    model = AutoModel.from_pretrained(repo)

    processor = AutoImageProcessor.from_pretrained(repo)

    return model, processor

def encode_image(model, processor, image):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.inference_mode():
        outputs = model(**inputs)

    cls_embeddings = outputs.pooler_output
    return cls_embeddings

# Functions to calculate distance
def pixelwise_distance(image1: Image.Image, image2: Image.Image) -> float:
    
    arr1 = np.array(image1)
    arr2 = np.array(image2)
    if arr1.shape != arr2.shape:
        raise ValueError("Images must have the same shape.")
    
    distance = np.linalg.norm(arr1 - arr2)
    distance = distance / np.sqrt(arr1.size)
    return distance

def latent_vector_distance(vector1: torch.Tensor, vector2: torch.Tensor) -> float:
    
    if vector1.shape != vector2.shape:
        raise ValueError("Latent vectors must have the same shape.")
    
    distance = torch.norm(vector1 - vector2)
    distance = distance / (vector1.numel() ** 0.5)

    return distance.item()


def main(args):

    seed_everything(42)

    if(args.num_shards > 0):
        assert args.shard is not None

    model = SiameseNetwork(network="ResNet-50", in_channels=3, n_features=128).to('cuda')

    # Loading ckpt
    CKPT_PATH = "trained_models/best_network.pth"
    # CKPT_PATH = "/raid/s2198939/diffusion_memorization/PatientVerification/checkpoints/ResNet-50_epoch1_data_handling_RPN.pth"
    model.load_state_dict(torch.load(CKPT_PATH))
    model.eval()

    CSV_DIR = "/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSV2"
    IMG_DIR = "/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
    train_csv = os.path.join(CSV_DIR, "FINAL_TRAIN.xlsx")
    test_csv = os.path.join(CSV_DIR, "FINAL_TEST.xlsx")
    val_csv = os.path.join(CSV_DIR, "FINAL_VAL.xlsx")

    print("Loading data...")
    df_train = pd.read_excel(train_csv)
    df_test = pd.read_excel(test_csv)
    df_val = pd.read_excel(val_csv)
    print("Data loaded successfully!")

    df_train["path"] = df_train["path"].apply(lambda x: os.path.join(IMG_DIR, x))
    df_test["path"] = df_test["path"].apply(lambda x: os.path.join(IMG_DIR, x))
    df_val["path"] = df_val["path"].apply(lambda x: os.path.join(IMG_DIR, x))

    # df_combined = pd.concat([df_train, df_test, df_val])

    df_mem_prompts = pd.read_csv("/raid/s2198939/diffusion_memorization/det_outputs_radedit/memorized_prompts_with_paths.csv")
    df_non_mem_prompts = pd.read_csv("/raid/s2198939/diffusion_memorization/det_outputs_radedit/non_memorized_prompts_with_paths.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if(args.run_on == "mem_prompts"):
        # mem_or_non_mem_paths = list(df_mem_prompts["path"].values)
        _df = df_mem_prompts
    elif(args.run_on == "non_mem_prompts"):
        # mem_or_non_mem_paths = list(df_non_mem_prompts["path"].values)
        _df = df_non_mem_prompts
    else:
        raise ValueError("Invalid value for --run_on. Choose between 'mem_prompts' and 'non_mem_prompts'.")
    
    if(args.num_shards > 0):
        print("Dividing into {} shards".format(args.num_shards))
        all_shards = np.array_split(_df, args.num_shards)
        _df = all_shards[args.shard].reset_index(drop=True)

    print("Length of dataframe: ", len(_df))

    NUM_GENERATIONS = 10
    # SEEDS = [42, 84, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
    SEEDS = random.choices(range(1, 1000), k=NUM_GENERATIONS)
    ALL_REAL_IMAGES_PATHS = []
    ALL_PROMPTS = []
    ALL_REID_SCORES = []
    ERROR_PROMPTS = []
    ALL_PIXEL_DISTANCES = []
    ALL_LATENT_DISTANCES = []
    

    # LOAD SD Pipeline (RadEdit)
    print("Loading SD Pipeline")
    pipe = load_sd_pipeline()
    print("Done!")

    print("Loading Encoding Model")
    encoding_model, processor = load_image_encoder()
    print("Done!")

    for i in tqdm(range(len(_df))):
        _PATH = _df['path'][i]
        _PROMPT = _df['text'][i]
        generated_images = []
        reid_scores = []
        pixel_distances = []
        latent_distances = []

        filename = _PATH.split("/")[-1].strip(".jpg")

        # Get real image from the training set corresponding to the prompt
        # There can be several real images corresponding to a prompt
        # Select 1
        
        try:
            real_img_path = df_train[df_train['text'] == _PROMPT].reset_index(drop=True)['path'][0]
            real_image = Image.open(real_img_path).resize((512, 512)).convert('RGB')

            print("Prompt: ", _PROMPT)
            # Generate images using this prompt
            for _seed in SEEDS:
                print("Generated with seed: ", _seed)
                gen_image = generate_synthetic_images(pipe, _PROMPT, 1, _seed)
                gen_image = gen_image.convert('RGB')
                generated_images.append(gen_image)

                # Save the generated image
                GEN_SAVE_DIR = "Output/Generated_Images/{}".format(args.run_on)
                os.makedirs(GEN_SAVE_DIR, exist_ok=True)

            # Calculate Re-id score between the real and the generated images
            print("Calculating Re-ID Scores!")
            for i, gen_img in enumerate(generated_images):

                # Save the generated image
                gen_img.save(os.path.join(GEN_SAVE_DIR, filename+"_gen_{}.jpg".format(i)))

                # Re-Identification Score
                reid_score = round(get_reidentification_score(model, real_image, gen_img, transform), 4)
                reid_scores.append(reid_score)

                # Pixel-Wise Distance
                pixel_dist = round(pixelwise_distance(real_image, gen_image), 4)
                pixel_distances.append(pixel_dist)

                # Latent Distance
                # 1. Encode the images
                real_img_encoded = encode_image(encoding_model, processor, real_image)
                syn_img_encoded = encode_image(encoding_model, processor, gen_img)
                latent_dist = round(latent_vector_distance(real_img_encoded, syn_img_encoded), 4)
                latent_distances.append(latent_dist)

            # Average out the scores for this mem/ non-mem image
            _avg_reid_score = sum(reid_scores) / len(reid_scores)
            ALL_REID_SCORES.append(_avg_reid_score)

            _avg_pixel_dist = sum(pixel_distances) / len(pixel_distances)
            ALL_PIXEL_DISTANCES.append(_avg_pixel_dist)

            _avg_latent_dist = sum(latent_distances) / len(latent_distances)
            ALL_LATENT_DISTANCES.append(_avg_latent_dist)
            
            ALL_REAL_IMAGES_PATHS.append(real_img_path)
            ALL_PROMPTS.append(_PROMPT)
            print("\n")
        
        except:
            print("No file was found for the prompt: ", _PROMPT)
            ERROR_PROMPTS.append(_PROMPT)
            continue

        # for img_path in all_images_paths:
        # for j in tqdm(range(len(all_images_paths))):
        #     img_path = all_images_paths[j]
        #     img2 = Image.open(img_path).convert('RGB')
        #     reid_score = get_reidentification_score(model, img1, img2, transform)
        #     reid_score = round(reid_score, 4)
        #     reid_scores.append(reid_score)
        #     all_paths1.append(_PATH)
        #     all_paths2.append(img_path)

    print("Mean ReID score: ", sum(ALL_REID_SCORES) / len(ALL_REID_SCORES))
    print("Mean Pixel Distances: ", sum(ALL_PIXEL_DISTANCES)/ len(ALL_PIXEL_DISTANCES))
    print("Mean Latent Distances: ", sum(ALL_LATENT_DISTANCES)/ len(ALL_LATENT_DISTANCES))

    # Create a dataframe of results of all scores
    try:
        results_df = pd.DataFrame()
        results_df['Real Path'] = ALL_REAL_IMAGES_PATHS
        results_df['Prompt'] = ALL_PROMPTS
        results_df["reid_scores"] = ALL_REID_SCORES
        results_df["Pixel_distance"] = ALL_PIXEL_DISTANCES
        results_df["Latext_distance"] = ALL_LATENT_DISTANCES

        results_name = f"reid_scores_{args.run_on}.csv" if args.num_shards == 0 else f"reid_scores_{args.run_on}_shard_{args.shard}.csv"
        errors_name = f"error_{args.run_on}.csv" if args.num_shards == 0 else f"error_{args.run_on}_shard_{args.shard}.csv"
        results_df.to_csv(results_name, index=False)

        if(len(ERROR_PROMPTS) > 0):
            errors_df = pd.DataFrame()
            errors_df['Error_Prompts'] = ERROR_PROMPTS
            errors_df.to_csv(errors_name, index=False)
    except:
        import pdb; pdb.set_trace()

if __name__ == "__main__":
    args = parse_args()
    main(args)
