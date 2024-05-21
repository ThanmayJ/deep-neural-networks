import torch
from tqdm import tqdm

import json
import os
import time

torch.manual_seed(42)
scaler = torch.cuda.amp.GradScaler()

def get_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    return round((correct / labels.size(0))*100,4)


def train(args, model, device, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for (images,labels) in tqdm(loader):
        images = images.to(device) # images.shape = (batch_size, in_channels, in_height, in_width)
        labels = labels.to(device) # labels.shape = (batch_size,)
        optimizer.zero_grad()

        if args.use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(images) # outputs.shape = (batch_size, out_classes)
                predictions = torch.argmax(outputs,dim=-1) # predictions.shape = (batch_size,)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward() # loss.backward()
            scaler.step(optimizer) #optimizer.step()
            scaler.update()
        
        else:
            outputs = model(images) # outputs.shape = (batch_size, out_classes)
            predictions = torch.argmax(outputs,dim=-1) # predictions.shape = (batch_size,)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()


        epoch_loss+=loss.item()
    
    return epoch_loss/len(loader)

def validate(args, model, device, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for (images,labels) in tqdm(loader):
            images = images.to(device) # images.shape = (batch_size, in_channels, in_height, in_width)
            labels = labels.to(device) # labels.shape = (batch_size,)
    
            
            if args.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(images) # outputs.shape = (batch_size, out_classes)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, labels)
            epoch_loss+=loss.item()
    
    return epoch_loss/len(loader)


def test(args, model, device, loader, criterion):
    model.eval()
    epoch_loss = 0
    predictions = torch.tensor([])
    labels = torch.tensor([])
    with torch.no_grad():
        for (_images,_labels) in tqdm(loader):
            _images = _images.to(device) # _images.shape = (batch_size, in_channels, in_height, in_width)
            _labels = _labels.to(device) # _labels.shape = (batch_size,)
    
            
            if args.use_fp16:
                with torch.cuda.amp.autocast():
                    _outputs = model(_images) # outputs.shape = (batch_size, out_classes)
            else:
                _outputs = model(_images)
                
            _predictions = torch.argmax(_outputs,dim=-1) # _predictions.shape = (batch_size,)
            loss = criterion(_outputs, _labels)

            
            predictions = torch.cat((predictions, _predictions.cpu()))
            labels = torch.cat((labels, _labels.cpu()))
            epoch_loss+=loss.item()
    return epoch_loss/len(loader), predictions, labels


def calculate_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_train_valid_maintain(dataset, train_ratio:float=0.8):
    num_classes = 10 # Assuming 10 classes. Change if required
    class2indices = dict(zip(list(range(10)),[[]]*num_classes)) # Maintains a list of indices for each class label

    print(f"Preparing train and validation splits by taking {train_ratio*100}% of train...")
    if not os.path.exists("../class2indices.json"):
        for idx, (image, label) in enumerate(dataset):
            class2indices[label].append(idx)
        with open('../class2indices.json', 'w') as f:
            json.dump(class2indices, f)
    else:
        print("Found exisiting split.")
        with open('../class2indices.json', 'r') as f:
            class2indices = json.load(f)

    train_indices = []
    valid_indices = []

    # Split the indices for each class into training and validation sets by maintaining the class distributions
    for class_idx in class2indices.values():
        num_samples = len(class_idx)
        num_train_samples = int(train_ratio * num_samples)
        
        train_indices.extend(class_idx[:num_train_samples])
        valid_indices.extend(class_idx[num_train_samples:])
    
    train_set = torch.utils.data.Subset(dataset, train_indices)
    valid_set = torch.utils.data.Subset(dataset, valid_indices)
    
    return train_set, valid_set

def split_train_valid_random(dataset, train_ratio:float=0.8):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    valid_size = dataset_size - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    
    return train_set, valid_set
