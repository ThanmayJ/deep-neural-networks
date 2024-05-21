"""
This script is required for questions 3 and 5. It takes a pretrained model, evaluates on the test set, and commits the WORD accuracy to wandb.
It also generates an optional attention heatmap (arg:attn_heatmap) for question 5d.
"""

from numpy import uint8

import csv

import argparse
import wandb

import torch

import time

from Model import VanillaSeq2SeqRNN, BahdanauSeq2SeqRNN
from Dataset import Aksharantar, AksharantarTokenizer
from utils import test, get_accuracy, count_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def evaluate(args, model, device, test_loader, criterion):
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    wandb.run.name = f"rnn:question5"

    print("[Test]")
    print("Number of parameters:",count_parameters(model))
    if args.attn_heatmap:
        assert args.use_attention, "Can't generate attention heatmap without using attention model. Specify use_attention hyperparam and try again."
        test_loss, predictions, labels, attention_matrix = test(args, model, device, test_loader, criterion, True)
    else:
        test_loss, predictions, labels = test(args, model, device, test_loader, criterion)
    predictions = tgt_tokenizer.decode_sequences(predictions)
    labels = tgt_tokenizer.decode_sequences(labels)
    word_acc, char_acc = get_accuracy(predictions,labels)

    filename = "predictions_attention.csv" if args.use_attention else "predictions_vanilla.csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Prediction', 'Label'])
        
        # Write the data
        for prediction, label in zip(predictions, labels):
            writer.writerow([prediction, label])
    
    print(f"Test Accuracy:{word_acc}\n")
    wandb.log({"word_accuracy":word_acc}, commit=True)

    if args.attn_heatmap:
        attention_matrix_normalized = (255 * (attention_matrix - torch.min(attention_matrix)))

        # Create an heatmap from the attention matrix
        attention_images = torch.stack([attention_matrix_normalized] * 3, dim=-1)  # Create an RGB image

        # Convert the heatmap to np.uint8
        attention_images = attention_images.cpu().detach().numpy().astype(uint8)

        # Log the image to wandb
        wandb.log({"question5:attention_heatmap": 
                   [wandb.Image(attention_image, caption=f"Attention Heatmap {predictions[-i]}/{labels[-i]}") 
                    for i,attention_image in enumerate(attention_images[-10:])]}, commit=True)

        # Finish the wandb run
        wandb.finish()



if __name__ == "__main__":
    print("Running on device", DEVICE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default="cs6910_assignment3",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument('--wandb_entity', type=str, default="thanmay",
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                        help="Learning rate to be used with Adam optimizer.")
    parser.add_argument('--epochs', type=int, default=20,
                        help="Number of epochs to train neural network.")
    parser.add_argument('--rnn_type', type=str, default='RNN', choices=['RNN', 'LSTM', 'GRU'],
                        help="Type of RNN to be used.")
    parser.add_argument('--dropout', type=float, default=0,
                        help="Number of epochs to train neural network.")
    parser.add_argument('--bidirectional', type=int, default=0,
                        help="Bidirectional or not")
    parser.add_argument('--num_layers', type=int, default=3,
                        help="Number of layers in encoder and decoder.")
    parser.add_argument('--dim_state', type=int, default=256,
                        help="Hidden state dimension of encoder and decoder")
    parser.add_argument('--dim_embed', type=int, default=256,
                        help="Embedding dimension of encoder and decoder")
    parser.add_argument('--max_len', type=int, default=32,
                        help="Max decoding length")
    parser.add_argument('--no_wandb', default=False, action='store_true',
                        help="Disable WandB if set True")
    parser.add_argument('--use_fp16', default=False, action='store_true',
                        help="Use fp16 with mixed precision")
    parser.add_argument('--data_dir', type=str, default="../aksharantar_sampled/",
                        help="Root directory of dataset.")
    parser.add_argument('--lang', type=str, default="hin",
                        help="Specify the target indic language. (source language is english)")
    parser.add_argument('--save_best_model', default=False, action="store_true",
                        help="If set, save_best_model is saved.")
    parser.add_argument('--use_attention', default=False, action="store_true",
                        help="If set, BahdanauAttention will be used between Encoder and Decoder.")
    parser.add_argument('--attn_heatmap', default=False, action='store_true',
                        help="If set, will plot the attention heatmap of (ensure args.use_attention is True)")
    parser.add_argument('--pretrained', type=str, default="results/",
                        help="Provide directory containing model.pth and optimizer.pth.")
    
    
        
    args = parser.parse_args()

    train_set = Aksharantar(args.data_dir, args.lang, "train")
    valid_set = Aksharantar(args.data_dir, args.lang, "valid")
    test_set = Aksharantar(args.data_dir, args.lang, "test")

    src_tokenizer = AksharantarTokenizer(train_set.bitext["src"]+valid_set.bitext["src"]+test_set.bitext["src"], bos_token="+", eos_token=".", pad_token=",")
    tgt_tokenizer = AksharantarTokenizer(train_set.bitext["tgt"]+valid_set.bitext["tgt"]+test_set.bitext["tgt"], bos_token="+", eos_token=".", pad_token=",")
    
    test_set.tokenize(src_tokenizer, tgt_tokenizer)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_set.collate_fn, shuffle=False)
    
    if args.use_attention:
        print("Using Bahdanau Seq2Seq")
        model = BahdanauSeq2SeqRNN(enc_dim_vocab = len(src_tokenizer),
                        enc_dim_embed = args.dim_embed,
                        enc_dim_state = args.dim_state,
                        enc_num_layers = args.num_layers,
                        dec_dim_vocab = len(tgt_tokenizer),
                        dec_dim_embed = args.dim_embed,
                        dec_dim_state = args.dim_state,
                        dec_num_layers = args.num_layers,
                        bidirectional = bool(args.bidirectional),
                        dropout = args.dropout,
                        rnn_type = args.rnn_type)
    else:
        print("Using Vanilla Seq2Seq")
        model = VanillaSeq2SeqRNN(enc_dim_vocab = len(src_tokenizer),
                        enc_dim_embed = args.dim_embed,
                        enc_dim_state = args.dim_state,
                        enc_num_layers = args.num_layers,
                        dec_dim_vocab = len(tgt_tokenizer),
                        dec_dim_embed = args.dim_embed,
                        dec_dim_state = args.dim_state,
                        dec_num_layers = args.num_layers,
                        bidirectional = bool(args.bidirectional),
                        dropout = args.dropout,
                        rnn_type = args.rnn_type)
    if args.pretrained is not None:
        model.load_state_dict(torch.load(args.pretrained+'/model.pth'))
    
    model = model.to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    evaluate(args, model, DEVICE, test_loader, criterion)
