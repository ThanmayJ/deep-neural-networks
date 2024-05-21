import torch
from tqdm import tqdm

import json
import os
import time

torch.manual_seed(42)
scaler = torch.cuda.amp.GradScaler()

def get_accuracy(predicted_words, label_words):
    print(predicted_words[:5])
    print(label_words[:5])
    assert len(predicted_words) == len(label_words), "The lists of predicted words and label words must have the same length."

    total_word_accuracy = 0
    total_char_accuracy = 0
    total_words = len(label_words)
    total_chars = 0
    correct_char_count = 0
    total_char_count = 0

    for pred_word, label_word in zip(predicted_words, label_words):
        # Word-level accuracy
        if pred_word == label_word:
            total_word_accuracy += 1

        # Character-level accuracy
        num_chars = len(label_word)
        if len(pred_word) <= num_chars:
            pred_word += ","*(num_chars-len(pred_word))
        total_char_count += num_chars
        correct_chars = sum(pc == lc for pc, lc in zip(pred_word, label_word))
        correct_char_count += correct_chars

    # Calculate average accuracies
    average_word_accuracy = total_word_accuracy / total_words if total_words > 0 else 0
    average_char_accuracy = correct_char_count / total_char_count if total_char_count > 0 else 0

    return average_word_accuracy, average_char_accuracy



def train(args, model, device, loader, optimizer, criterion, teacher_forcing):
    model.train()
    epoch_loss = 0

    pad_token_id = loader.dataset.special_tokens["tgt"]["pad"]["id"]
    bos_token_id = loader.dataset.special_tokens["tgt"]["bos"]["id"]
    eos_token_id = loader.dataset.special_tokens["tgt"]["eos"]["id"]

    for (src_tokens, tgt_tokens) in tqdm(loader):
        src_tokens = src_tokens.to(device) # src_tokens.shape = (batch_size, seq_len)
        tgt_tokens = tgt_tokens.to(device) # tgt_tokens.shape = (batch_size, seq_len)
        optimizer.zero_grad()

        if args.use_fp16:
            with torch.cuda.amp.autocast():
                outputs = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing) # outputs.shape = (batch_size, seq_len, vocab_size)
                pred_tokens = torch.argmax(outputs, dim=-1) # pred_tokens.shape = (batch_size, seq_len)
                loss = criterion(outputs, tgt_tokens)

            scaler.scale(loss).backward() # loss.backward()
            scaler.step(optimizer) #optimizer.step()
            scaler.update()
        
        else:
            outputs = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing) # outputs.shape = (batch_size, seq_len, vocab_size)
            pred_tokens = torch.argmax(outputs, dim=-1) # pred_tokens.shape = (batch_size, seq_len)
            outputs, tgt_tokens = model.pad_outputs(outputs, tgt_tokens, pad_token_id)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_tokens.view(-1))
            
            loss.backward()
            optimizer.step()


        epoch_loss+=loss.item()
    
    return epoch_loss/len(loader)

def validate(args, model, device, loader, criterion):
    model.eval()
    epoch_loss = 0

    pad_token_id = loader.dataset.special_tokens["tgt"]["pad"]["id"]
    bos_token_id = loader.dataset.special_tokens["tgt"]["bos"]["id"]
    eos_token_id = loader.dataset.special_tokens["tgt"]["eos"]["id"]

    with torch.no_grad():
        for (src_tokens, tgt_tokens) in tqdm(loader):
            src_tokens = src_tokens.to(device) # src_tokens.shape = (batch_size, seq_len)
            tgt_tokens = tgt_tokens.to(device) # tgt_tokens.shape = (batch_size, seq_len)
    
            
            if args.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing=0) # outputs.shape = (batch_size, seq_len, vocab_size)
                    outputs, tgt_tokens = model.pad_outputs(outputs, tgt_tokens, pad_token_id)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_tokens.view(-1))
            else:
                outputs = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing=0) # outputs.shape = (batch_size, seq_len, vocab_size)
                outputs, tgt_tokens = model.pad_outputs(outputs, tgt_tokens, pad_token_id)
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_tokens.view(-1))
            
            epoch_loss+=loss.item()
    
    return epoch_loss/len(loader)


def test(args, model, device, loader, criterion, return_attn_heatmap=False):
    model.eval()
    epoch_loss = 0

    pad_token_id = loader.dataset.special_tokens["tgt"]["pad"]["id"]
    bos_token_id = loader.dataset.special_tokens["tgt"]["bos"]["id"]
    eos_token_id = loader.dataset.special_tokens["tgt"]["eos"]["id"]

    predictions = []
    references = []
    with torch.no_grad():
        for (src_tokens, tgt_tokens) in tqdm(loader):
            src_tokens = src_tokens.to(device) # src_tokens.shape = (batch_size, seq_len)
            tgt_tokens = tgt_tokens.to(device) # tgt_tokens.shape = (batch_size, seq_len)
    
            
            if args.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing=0) # outputs.shape = (batch_size, seq_len, vocab_size)
                    pred_tokens = torch.argmax(outputs, dim=-1) # pred_tokens.shape = (batch_size, seq_len)
                    outputs, tgt_tokens = model.pad_outputs(outputs, tgt_tokens, pad_token_id)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_tokens.view(-1))
            else:
                if return_attn_heatmap:
                    outputs, attn_map = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing=0, return_attn_weights=return_attn_heatmap) # outputs.shape = (batch_size, seq_len, vocab_size)
                else:
                    outputs = model(src_tokens, args.max_len, bos_token_id, eos_token_id, tgt_tokens, teacher_forcing=0) # outputs.shape = (batch_size, seq_len, vocab_size)
                pred_tokens = torch.argmax(outputs, dim=-1) # pred_tokens.shape = (batch_size, seq_len)
                outputs, tgt_tokens = model.pad_outputs(outputs, tgt_tokens, pad_token_id)
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_tokens.view(-1))

            
            predictions.extend(pred_tokens.cpu().detach().tolist())
            references.extend(tgt_tokens.cpu().detach().tolist())
            epoch_loss+=loss.item()
        
    if return_attn_heatmap:
        return epoch_loss/len(loader), predictions, references, attn_map
    else:
        return epoch_loss/len(loader), predictions, references


def calculate_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
