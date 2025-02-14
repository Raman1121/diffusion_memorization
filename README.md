# The Devil is in the Prompts: De-Identification Traces Enhance Memorization Risks in Synthetic Chest X-Ray Generation
<!-- Official repo for [The Devil is in the Prompts: De-Identification Traces Enhance Memorization Risks in Synthetic Chest X-Ray Generation](https://arxiv.org/abs/2502.07516). -->

## Core Dependencies
- PyTorch==2.5.1
- transformers==4.48.2
- diffusers==0.18.2
- accelerate==0.21.0
- datasets==2.19.0

You can create the environment using:  
```
conda env create -f env.yaml
```

## Memorized Prompts
The list of memorized prompts in the MIMIC-CXR dataset will be available here.

## Detecting Memorization

### Setting up the Dataset and DataLoaders
First, create a standard pytorch dataset (For MIMIC-CXR, see the ```mimic_cxr_dataset.py``` file.)  
Add the data loading logic in the ```get_dataset``` function within the ```optim_utils.py``` file.

### Using (any) Stable Diffusion Model
```
python detect_mem.py --model_id <pretrained_model_name_or_path> --dataset mimic
```

### Using the RadEdit Model

***NOTE:*** First, follow the instructions in the [Setup-RadEdit.md](Setup-RadEdit.md) file.  

The following command will run memorization detection on the whole dataset.

```
python detect_mem_radedit.py --run_name memorized_prompts --dataset mimic
```

***NOTE:*** Due to the large size of MIMIC-CXR, it is recommended to run the detection on smaller subsets (shards) of the dataset in parallel. 

The following command will divide the dataset into ```4``` shards. You now need to run ```4``` scripts in parallel and only change the ```--shard``` parameter in the following way:

```
python detect_mem_radedit.py --run_name memorized_prompts --dataset mimic --num_shards 4 --shard 0
python detect_mem_radedit.py --run_name memorized_prompts --dataset mimic --num_shards 4 --shard 1
python detect_mem_radedit.py --run_name memorized_prompts --dataset mimic --num_shards 4 --shard 2 
python detect_mem_radedit.py --run_name memorized_prompts --dataset mimic --num_shards 4 --shard 3
```

For even faster computation, you can increase ```--num_shards```.

<!-- ## Cite our work
If you find this work useful, please cite our paper:

```bibtex
@article{dutt2025devil,
  title={The Devil is in the Prompts: De-Identification Traces Enhance Memorization Risks in Synthetic Chest X-Ray Generation},
  author={Dutt, Raman},
  journal={arXiv preprint arXiv:2502.07516},
  year={2025}
}
``` -->
