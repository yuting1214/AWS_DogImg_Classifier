#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import argparse
import os
import sys
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, device, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    test_loss = 0
    correct = 0
    dataloader = test_loader['test']
    dataset_sizes = len(dataloader.dataset)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)
    test_loss /= dataset_sizes
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, dataset_sizes, 100.0 * correct / dataset_sizes
            )
        )

def train(model, device, dataloaders, criterion, optimizer, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(1, args.num_epochs + 1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            dataloader = dataloaders[phase] # Class: torch.utils.data.dataloader.DataLoade
            dataset_sizes = len(dataloader.dataset)
            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects.double() / dataset_sizes
            if phase == 'train':
                train_epoch_loss = epoch_loss
                train_epoch_acc = epoch_acc
            else:
                val_epoch_loss = epoch_loss
                val_epoch_acc = epoch_acc
        logger.info(
            "Epoch: {}/{}{}Train Loss: {:.4f} Acc: {:.4f}{}Val Loss: {:.4f} Acc: {:.4f}{}{}".format(
                epoch,
                args.num_epochs,
                '\n',
                train_epoch_loss,
                train_epoch_acc,
                '\n',
                val_epoch_loss,
                val_epoch_acc,
                '\n',
                '----------'))
    return model

def net(num_output_class, device):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model_conv = models.resnet18(pretrained=True)
    # for param in model_conv.parameters():
    #      param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_output_class)
    return model_conv.to(device)

def create_data_loaders(data_type, batch_size, data_dir):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    '''
    Data preprocessing
    '''
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    assert data_type in ['train', 'valid', 'test']
    if data_type == 'train' or data_type == 'valid':
        type_list = ['train', 'valid']
    else:
        type_list = ['test']
    logger.info(f"Get {data_type} data_loader")
    dataloaders = {x: torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]),
                                                    batch_size=batch_size, shuffle=True, num_workers=4) for x in type_list}
    return dataloaders

def main(args):

    '''
    TODO: Initialize data_loaders
    '''
    train_loader = create_data_loaders('train', args.batch_size, args.data_dir)
    test_loader = create_data_loaders('test', args.test_batch_size, args.data_dir)
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_names = datasets.ImageFolder(os.path.join(args.data_dir, 'train')).classes
    num_output_class = len(class_names)
    model = net(num_output_class, device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model = train(model, device, train_loader, loss_criterion, optimizer, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, device, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pt')
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"]) # Retre8ve from output_path in Estimator
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]) # Retreive from tuner.fit({"training": inputs})
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    args=parser.parse_args()    
    main(args)
