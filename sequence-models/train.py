import argparse
import wandb

import torch

import time

from Model import VanillaSeq2SeqRNN, BahdanauSeq2SeqRNN
from Dataset import Aksharantar, AksharantarTokenizer
from utils import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def Trainer(args, model, device, dataloaders, optimizer, criterion, src_tokenizer, tgt_tokenizer):
    log_wandb = not(args.no_wandb)
    if log_wandb:
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = f"sweep2:Vanilla{args.rnn_type}|bidirectional:{bool(args.bidirectional)}|dim_state-{args.dim_state}|num_layers-{args.num_layers}|learning_rate-{args.learning_rate}|dropout-{args.dropout}" # Replace with an apt name to identify the run using the command-line arguments (args)

    best_test_accuracy = 0
    best_val_loss = float('inf')
    train_start = time.time()
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"[Epoch {epoch}]")
        teacher_forcing_decrement = 1 / args.epochs
        teacher_forcing = 1 - teacher_forcing_decrement * (epoch)
        if teacher_forcing < 0:
            teacher_forcing = 0
        print("Teacher forcing is:",teacher_forcing)
        train_loss = train(args, model, device, dataloaders["train"], optimizer, criterion, teacher_forcing)
        end_time = time.time()
        epoch_mins, epoch_secs = calculate_time(start_time, end_time)
        print(f"Train Time: {epoch_mins}m {epoch_secs}s | Train Loss {train_loss}")

        print("[Validation]")
        val_loss, predictions, labels = test(args, model, device, dataloaders["validation"], criterion)
        predictions = tgt_tokenizer.decode_sequences(predictions)
        labels = tgt_tokenizer.decode_sequences(labels)
        val_accuracy = get_accuracy(predictions,labels)
        print(f"Validation Accuracy: {val_accuracy} | Validataion Loss: {val_loss}")

        print("[Test]")
        test_loss, predictions, labels = test(args, model, device, dataloaders["test"], criterion)
        predictions = tgt_tokenizer.decode_sequences(predictions)
        labels = tgt_tokenizer.decode_sequences(labels)
        test_accuracy = get_accuracy(predictions,labels)
        print(f"Test Accuracy:{test_accuracy}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_accuracy = test_accuracy
            if args.save_best_model:
                torch.save(model.state_dict(), 'results/model.pth')
                torch.save(optimizer.state_dict(), 'results/optimizer.pth')

        if log_wandb:
            wandb.log({"loss": train_loss, "val_accuracy":val_accuracy, "val_loss":val_loss, "word_acc":val_accuracy[0], "char_acc":val_accuracy[1], "epoch":epoch}, step=epoch, commit=False)
    
    if log_wandb:
        wandb.log({"best_test_accuracy":best_test_accuracy}, commit=True)
    train_end = time.time()
    train_mins, train_secs = calculate_time(train_start, train_end)
    print(f"""Total Training Time was {train_mins} m {train_secs} s for {args.epochs} epochs""")    


    

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
                        help="Type of RNN to be used")
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
    
    
        
    args = parser.parse_args()

    train_set = Aksharantar(args.data_dir, args.lang, "train")
    valid_set = Aksharantar(args.data_dir, args.lang, "valid")
    test_set = Aksharantar(args.data_dir, args.lang, "test")

    src_tokenizer = AksharantarTokenizer(train_set.bitext["src"]+valid_set.bitext["src"]+test_set.bitext["src"], bos_token="+", eos_token=".", pad_token=",")
    tgt_tokenizer = AksharantarTokenizer(train_set.bitext["tgt"]+valid_set.bitext["tgt"]+test_set.bitext["tgt"], bos_token="+", eos_token=".", pad_token=",")
    
    train_set.tokenize(src_tokenizer, tgt_tokenizer)
    valid_set.tokenize(src_tokenizer, tgt_tokenizer)
    test_set.tokenize(src_tokenizer, tgt_tokenizer)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=test_set.collate_fn, shuffle=False)
    dataloaders = {"train":train_loader, "validation":valid_loader, "test":test_loader}
    
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

    
    # print(count_parameters(model))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    model = model.to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_loader.dataset.special_tokens["tgt"]["pad"]["id"])
    
    Trainer(args, model, DEVICE, dataloaders, optimizer, criterion, src_tokenizer, tgt_tokenizer)