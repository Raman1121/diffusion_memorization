## Instructions for Detecting Memorization with RadEdit

***NOTE:*** These instuctions are also listed on top of the ```detect_mem_radedit.py``` file.

You will need to make some changes in the installed ```diffusers``` package to solve some dependency issues.

1. Make sure you have created the conda environment. Assuming the name of your conda environment is `demm2`.

2. Navigate to `/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/__init__.py` and **comment** out in the following way:
   
   ```python
   from .pipelines import (
       AudioPipelineOutput,
       # ConsistencyModelPipeline,
       # DanceDiffusionPipeline,
       ...
   )

3. Navigate to `/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/pipelines/pipeline_utils.py` and comment out in the following way:
```from transformers.utils import SAFE_WEIGHTS_NAME as TRANSFORMERS_SAFE_WEIGHTS_NAME```  

4. Navigate to `/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/pipelines/__init__.py` and comment out in the following way:
```# from .consistency_models import ConsistencyModelPipeline```
```# from .dance_diffusion import DanceDiffusionPipeline```

5. Navigate to ```/miniconda3/envs/demm2/lib/python3.10/site-packages/diffusers/pipelines/consistency_models/__init__.py``` and comment out in the following way:
```# from .pipeline_consistency_models import ConsistencyModelPipeline```

