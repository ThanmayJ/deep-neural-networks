import argparse
import wandb

import torch
from torchvision import transforms, datasets

from tqdm import tqdm
import time

from Model import ConvNet
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def Trainer(args, model, device, dataloaders, optimizer, criterion):
    log_wandb = not(args.no_wandb)
    if log_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = f"partA" # Replace with an apt name to identify the run using the command-line arguments (args)

    best_test_accuracy = 0
    best_val_loss = float('inf')
    train_start = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"[Epoch {epoch}]")
        train_loss = train(args, model, device, dataloaders["train"], optimizer, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = calculate_time(start_time, end_time)
        print(f"Train Time: {epoch_mins}m {epoch_secs}s | Train Loss {train_loss}")

        print("[Validation]")
        val_loss, predictions, labels = test(args, model, device, dataloaders["validation"], criterion)
        val_accuracy = get_accuracy(predictions,labels)
        print(f"Validation Accuracy: {val_accuracy} | Validataion Loss: {val_loss}")

        print("[Test]")
        test_loss, predictions, labels = test(args, model, device, dataloaders["test"], criterion)
        test_accuracy = get_accuracy(predictions,labels)
        print(f"Test Accuracy:{test_accuracy}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_accuracy = test_accuracy
            if args.save_best_model:
                torch.save(model.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')

        if log_wandb:
            wandb.log({"loss": train_loss, "val_accuracy":val_accuracy, "val_loss":val_loss, "epoch":epoch}, step=epoch, commit=False)
    
    if log_wandb:
        wandb.log({"best_test_accuracy":best_test_accuracy}, commit=True)
    train_end = time.time()
    train_mins, train_secs = calculate_time(train_start, train_end)
    print(f"""Total Training Time was {train_mins} m {train_secs} s for {args.epochs} epochs""")    


    

if __name__ == "__main__":
    print("Running on device", DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default="cs6910_assignment2",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('--wandb_entity', type=str, default="thanmay",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument('--in_height', type=int, default=224,
                        help="Height of the input image in pixels.")
    parser.add_argument('--in_width', type=int, default=224,
                        help="Width of the input image in pixels.")
    parser.add_argument('--in_channels', type=int, default=3,
                        help="Number of channels in the input image.")
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="Number of epochs to train neural network.")
    parser.add_argument('--filters', type=int, default=70,
                        help="Number of filters to be used per layer.")
    parser.add_argument('--num_layers', type=int, default=5,
                        help="Number convolution layers in the network.")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Number of output classes.")
    parser.add_argument('--dense_features', type=int, default=512,
                        help="Number neurons in the dense layer.")
    parser.add_argument('--kernel_size', type=int, default=7,
                        help="Size of the (square) filter to be used in convolution and pooling.")
    parser.add_argument('--stride', type=int, default=1,
                        help="Stride to be used in convolution and pooling.")
    parser.add_argument('--padding', type=int, default=0,
                        help="Padding to be used in convolution and pooling.")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout to be used after dense layer. Set 0 if no dropout.")
    parser.add_argument('--use_batchnorm', default=False, action="store_true",
                        help="Set if use batchnorm after convolution and dense layer.")
    parser.add_argument('--augmentation', default=False, action="store_true",
                        help="Set if use data augmentation.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate to be used with Adam optimizer.")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of epochs to train neural network.")
    parser.add_argument('--no_wandb', default=False, action='store_true',
                        help="Disable WandB if set True")
    parser.add_argument('--use_fp16', default=False, action='store_true',
                        help="Use fp16 with mixed precision")
    parser.add_argument('--data_dir', type=str, default="../../nature_12K/inaturalist_12K",
                        help="Root directory of dataset. Ensure subfolders are train/ and val/")
    parser.add_argument('--maintain_dist_after_split', default=False, action="store_true",
                        help="Set if class distributions are to be maintained when splitting train and validation. Else random split.")
    parser.add_argument('--manual_layers',type=str,
                        help="Used to provide number of filters per layer manually (eg. '10,30,70,90' ie. 4 conv layers with these number of filters in order). If provided, will override --num_layers and --kernel_size")
    parser.add_argument('--save_best_model', default=False, action="store_true",
                        help="Set if model with best validation loss is to be saved after training.")
    parser.add_argument('--pretrained', type=str, default=None,
                        help="Provide directory containing model.pth and optimizer.pth.")
    
    
        
    args = parser.parse_args()

    if args.manual_layers is not None:
        args.manual_layers = [int(x) for x in args.manual_layers.split(",")]
        layer_filters_list = args.manual_layers
    else:
        layer_filters_list = [args.filters] * args.num_layers
    
    transform = [transforms.Resize((args.in_height, args.in_width)),
                 transforms.ToTensor()]
    if args.augmentation:
        transform += [transforms.RandomRotation(degrees=30),
            transforms.RandomResizedCrop(size=(args.in_height, args.in_width), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1)]
    
    transform = transforms.Compose(transform)
    
    
    train_set = datasets.ImageFolder(root=args.data_dir+'/train/', transform=transform)
    if args.maintain_dist_after_split:
        train_set, valid_set = split_train_valid_maintain(train_set, train_ratio=0.8)
    else:
        train_set, valid_set = split_train_valid_random(train_set, train_ratio=0.8)
    
    test_set = datasets.ImageFolder(root=args.data_dir+'/val/', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    dataloaders = {"train":train_loader, "validation":valid_loader, "test":test_loader}
    
    model = ConvNet(in_height=args.in_height, in_width=args.in_width, in_channels=args.in_channels, activation=args.activation, layer_filters_list=layer_filters_list, dense_features=args.dense_features, out_classes=args.num_classes, kernel_size=args.kernel_size, stride=args.stride, padding=args.padding, dropout=args.dropout, use_batchnorm=args.use_batchnorm)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained+'/model.pth'))
        optimizer.load_state_dict(torch.load(args.pretrained+'/optimizer.pth'))
    
    model = model.to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    Trainer(args, model, DEVICE, dataloaders, optimizer, criterion)