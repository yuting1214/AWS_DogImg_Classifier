# Dog breeds image classification

Use transfered learning with pretrained model of Resnet18 and Resnet50 in Pytorch Framework to conduct a image classification task of classifying 133 dog breeds.

## Project Set Up and Installation
**OPTIONAL:** If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to make your `README` detailed and self-explanatory. For instance, here you could explain how to set up your project in AWS and provide helpful screenshots of the process.

## Dataset

### Overview
**TODO**: Explain about the data you are using and where you got it from.
Come from Dog Breed Identification dataset(https://www.kaggle.com/competitions/dog-breed-identification/data)
Now the dataset I used has 133 breeds.

```
Issue1
Error: OSError: image file is truncated (150 bytes not processed)

Cause: PIl fails to load the image(dogImages\train\Leonberger_06571.jpg).

Solution: Create a new image by PIL

Sources:
https://discuss.pytorch.org/t/oserror-image-file-is-truncated-28-bytes-not-processed-during-learning/36815/21
```

try:
with Image.open(os.path.normpath(r’C:\Users\l501l\Desktop\Leonberger_06571.jpg’)) as im:
im.save(os.path.normpath(r’C:\Users\l501l\Desktop\Leonberger_06571_new.jpg’))
except OSError:
print(“cannot convert”, infile)

### Access
**TODO**: Explain how you are accessing the data in AWS and how you uploaded it
Upload the data into S3 by AWS console with three folders including train, test, and valid.

## Hyperparameter Tuning
**TODO**: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Use both ResNet18 and ResNet50 for transfer learning, since ResNet has strong edge to achieve higher accuracy in network performance especially in the application of Image Classification.

The hyperparamter used in model are learning rate and batch size with the range between 


Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

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
