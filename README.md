# End-to-end Dog breeds image classifier in AWS

Established an end-to-end data API service for image classification in AWS, utilizing transfer learning with pre-trained models in PyTorch for a task involving the classification of 133 dog breeds.

:question: **Problem:** How to establish an end-to-end data API service for image classification in AWS.

:key: **Importance:** Demonstrate the ability to deploy a data solution by integrating data engineering skills with advanced deep learning techniques.

🎉 **Achievements:** Implemented transfer learning with pre-trained models to accurately classify a diverse set of 133 dog breeds in 90% accuracy, and established an efficient API service in AWS for seamless integration with other applications.

💪 **Skills:** Python, Pytorch, AWS SageMaker, Deep learning

:books: **Insights:** The cost-effective way to train, fine-tune, and deploy models in AWS cloud services.

## Project Set Up and Installation

### Files
```
1. train_and_deploy.ipynb: Use sagamaker for train and deploy model in Notebook instance
2. hpo.py: Script for hypeparameter tuning.
3. train_model.py: Script for training the final model
4. inference_final.py: Script for construct endpoint
```

### Packages preparation
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
Now the dataset used in this project has 133 breeds.

### Preprocessing
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
![ResNet18](https://github.com/yuting1214/Udacity_Proj3/blob/main/screenshot/Training_complete_resnet18.png)

![ResNet50](https://github.com/yuting1214/Udacity_Proj3/blob/main/screenshot/Training_complete_resnet50.png)

### Logs metrics:

Example of the logs metrics in one of the training job

![logs metrics](https://github.com/yuting1214/Udacity_Proj3/blob/main/screenshot/Log_metrics.png)

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

Take the profiler report from training ResNet18 model as an instance, since we're using fine-tuned mechanisum to update the weights of the neural network. The main comsumption of comptutaional resouces spent on updating the weights in the pretrained layers especially the CNN layers.

![Profiler Report](https://github.com/yuting1214/Udacity_Proj3/blob/main/screenshot/profiler.png)


## Model Deployment

### Overview

1. Use The [SageMaker PyTorch Model Server](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#id3)

2. Create inference.py for serializing prediction

3. Create an instance of "PyTorchModel" 

4. In PyTorchModel, specify the directory of inference.py as entry_point and the model artificat
 
5. Deploy an endpoint wtih EC2 instance of "ml.m5.xlarge" due to free-tier.

### Example
```
# 1. Provide the url of image

request_dict={ "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/26151212/Afghan-Hound-standing-in-a-garden.jpg"}
img_bytes = requests.get(request_dict['url']).content
Image.open(io.BytesIO(img_bytes))

# 2. Invocate the endpoint to make a prediction of the input
pred2 = predictor.predict(json.dumps(request_dict), initial_args={"ContentType": "application/json"})
pred2
# '002.Afghan_hound'
```

![Endpoint](https://github.com/yuting1214/Udacity_Proj3/blob/main/screenshot/endpoint.png)

![Example](https://github.com/yuting1214/Udacity_Proj3/blob/main/screenshot/dog_example.png)

## Standout Suggestions

Try deploy models with Docker when available.

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
```
# 1. Remember to shuffle the data when using dataloader
# 2. For some optimizer, like Adam, be sure to set lower learning rate(lr.)
```

