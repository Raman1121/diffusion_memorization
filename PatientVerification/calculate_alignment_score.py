from health_multimodal.text import get_bert_inference
from health_multimodal.image import get_image_inference
from health_multimodal.vlp import ImageTextInferenceEngine
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate ReID scores between memorized and non-memorized prompts.")
    parser.add_argument("--run_on", type=str, default="mem_prompts", choices=['mem_prompts', 'non_mem_prompts'], help="Choose between 'mem_prompts' and 'non_mem_prompts'.")
    # parser.add_argument("--num_shards", type=int, default=0, help="Number of shards.")
    # parser.add_argument("--shard", type=int, help="Shard index")
    return parser.parse_args()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def main(args):
    text_inference = get_bert_inference()
    image_inference = get_image_inference()
    image_text_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )

    # gen_cols = ['Gen 1 Score', 'Gen 2 Score', 'Gen 3 Score', 'Gen 4 Score', 'Gen 5 Score', 'Gen 6 Score', 'Gen 7 Score', 'Gen 8 Score', 'Gen 9 Score', 'Gen 10 Score']
    gen_cols = ['Gen {} Score'.format(_+1) for _ in range(10)]
    cols = ['Real Path', 'Real Img Score']
    cols.extend(gen_cols)
    results_df = pd.DataFrame(columns=cols)
    print(results_df)

    if(args.run_on == 'mem_prompts'):
        df = pd.read_csv("saved_shards/combined_mem_scores.csv")
    else:
        df = pd.read_csv("saved_shards/combined_non_mem_scores.csv")

    for i in tqdm(range(len(df))):
        row_data = []
        _path = df['Real Path'][i]
        _filename = _path.split("/")[-1]
        _gen_filenames = ['Output/Generated_Images/{}/'.format(args.run_on) + _filename.strip(".jpg") + '_gen_{}.jpg'.format(j) for j in range(10)]
        _prompt = df['Prompt'][i]
        real_img_score = round(image_text_inference.get_similarity_score_from_raw_data(Path(_path), _prompt), 3)
        
        row_data.append(_path)
        row_data.append(real_img_score)
        # Calculating scores for generated images
        gen_scores = []
        for _gen_path in _gen_filenames:
            gen_scores.append(
                round(image_text_inference.get_similarity_score_from_raw_data(
                    Path(_gen_path), 
                    _prompt
                ), 3)
            )
        print(gen_scores)
        row_data.extend(gen_scores)
        try:
            results_df.loc[len(results_df)] = row_data
        except:
            # import pdb; pdb.set_trace()
            continue

    print(results_df.head())
    results_df.to_csv("Alignment_scores_{}.csv".format(args.run_on), index=False)

if __name__ == "__main__":
    args = parse_args()
    main(args)