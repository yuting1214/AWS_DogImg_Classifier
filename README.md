# End-to-end Dog breeds image classification

Use transfered learning with pretrained model of Resnet18 and Resnet50 in Pytorch Framework to conduct a image classification task of classifying 133 dog breeds.

## Project Set Up and Installation

```
# For Notebook instance 
import sagemaker
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.pytorch import PyTorch

# For training script
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import argparse
import os
import sys
import logging
import json
from tqdm import tqdm

# For inference script
import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import argparse

```
## Dataset

### Overview

Come from [Dog Breed Identification dataset](https://www.kaggle.com/competitions/dog-breed-identification/data).
Now the dataset I used has 133 breeds.

```
Issue_1
Error: OSError: image file is truncated (150 bytes not processed)

Cause: PIl fails to load the image(dogImages\train\Leonberger_06571.jpg).

Solution: 

#Create a new image by PIL
try:
with Image.open(os.path.normpath(r'C:\Users\l501l\Desktop\Leonberger_06571.jpg')) as im:
    im.save(os.path.normpath(r'C:\Users\l501l\Desktop\Leonberger_06571_new.jpg'))
except OSError:
    print("cannot convert", infile)

Sources:
https://discuss.pytorch.org/t/oserror-image-file-is-truncated-28-bytes-not-processed-during-learning/36815/21
```

### Access
```
Upload the data into S3 by AWS console with three folders including train, test, and valid.

Directory: s3://udacity-mark/Proj3/dogImages/
```


## Hyperparameter Tuning
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Use both ResNet18 and ResNet50 for transfer learning, since ResNet has strong edges to achieve higher accuracy in network performance especially in the application of Image Classification.

The hyperparamters used in model are learning rate and batch size.

Use Bayesian Optimization to conduct hyperparameter-tuning with 4 hyperparameter combinations in each architecture.

```
# ResNet18
hyperparameter_ranges = {
    "lr": ContinuousParameter(0.001, 0.01),
    "batch-size": CategoricalParameter([32, 64]),
}
# ResNet50
hyperparameter_ranges = {
    "learning_rate": ContinuousParameter(0.001, 0.1),
    "batch_size": CategoricalParameter([32, 64, 128, 256, 512]),
}
```

### Completed training jobs:

### Logs metrics:

Example of the logs metrics in one of the training jobs.

### Best combinations of hyperparameters

```
{'_tuning_objective_metric': '"Test Loss"',
 'batch_size': '"128"',
 'learning_rate': '0.004338940031756185',
 'sagemaker_container_log_level': '20',
 'sagemaker_estimator_class_name': '"PyTorch"',
 'sagemaker_estimator_module': '"sagemaker.pytorch.estimator"',
 'sagemaker_job_name': '"Proj_4_pytorch_dog_hpo-2022-09-13-19-59-49-759"',
 'sagemaker_program': '"hpo.py"',
 'sagemaker_region': '"us-east-2"',
 'sagemaker_submit_directory': '"s3://sagemaker-us-east-2-800573291199/Proj_4_pytorch_dog_hpo-2022-09-13-19-59-49-759/source/sourcedir.tar.gz"'}
```

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

```
# For Debugger, check the following properties

rules = [
    Rule.sagemaker(rule_configs.vanishing_gradient()),
    Rule.sagemaker(rule_configs.overtraining()),
    Rule.sagemaker(rule_configs.loss_not_decreasing())
]
# For Profiler, render the Profiler Report

rules = [
    ProfilerRule.sagemaker(rule_configs.ProfilerReport())
]
```


### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?


**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.

## Note

### Environment setting
```
# 1. For supplying arguments for hyperparameter tuning
import argparse
parser=argparse.ArgumentParser()
## Model directory
parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]) # Retreive from Estimator(output_path = )
## Data directory
### For single location
parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]) # Retreive from tuner.fit({"training": inputs})
### For multiple directories
parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")) # Retreive from tuner.fit({"train": inputs}
parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST")) # # Retreive from tuner.fit({"test": inputs}

# 2. Model as hyparameter
hyperparameters = {
    ...
    "model": "resnext50_32x4d",
}
parser.add_argument("--model", type=str, default="resnet50")
net = models.__dict__[opt.model](pretrained=True)
```
### For training techniques
