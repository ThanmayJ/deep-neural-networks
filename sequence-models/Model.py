import torch
from torch import nn

from typing import Tuple, Optional
from torch import Tensor

import torch
import torch.nn as nn

import torch
import torch.nn as nn

import numpy as np

class BahdanauSeq2SeqRNN(nn.Module):
    def __init__(self, enc_dim_vocab, enc_dim_embed, enc_dim_state, enc_num_layers,
                 dec_dim_vocab, dec_dim_embed, dec_dim_state, dec_num_layers,
                 bidirectional=True, dropout=0.1, rnn_type='RNN'):
        super(BahdanauSeq2SeqRNN, self).__init__()

        # Encoder
        self.encoder_embedding = nn.Embedding(enc_dim_vocab, enc_dim_embed)
        if rnn_type == 'LSTM':
            self.encoder_rnn = nn.LSTM(enc_dim_embed, enc_dim_state, enc_num_layers, batch_first=True,
                                       bidirectional=bidirectional, dropout=dropout if dropout > 0 else 0)
        elif rnn_type == 'GRU':
            self.encoder_rnn = nn.GRU(enc_dim_embed, enc_dim_state, enc_num_layers, batch_first=True,
                                      bidirectional=bidirectional, dropout=dropout if dropout > 0 else 0)
        else:  # Default to RNN
            self.encoder_rnn = nn.RNN(enc_dim_embed, enc_dim_state, enc_num_layers, batch_first=True,
                                      bidirectional=bidirectional, dropout=dropout if dropout > 0 else 0)
        self.encoder_num_directions = 2 if bidirectional else 1

        # Attention
        self.attn = nn.Linear(enc_dim_state + dec_dim_state, dec_dim_state)
        self.attn_combine = nn.Linear(enc_dim_state + dec_dim_embed, dec_dim_embed)
        self.softmax = nn.Softmax(dim=1)

        # Decoder
        self.decoder_embedding = nn.Embedding(dec_dim_vocab, dec_dim_embed)
        if rnn_type == 'LSTM':
            self.decoder_rnn = nn.LSTM(dec_dim_embed, dec_dim_state, dec_num_layers, batch_first=True,
                                       bidirectional=False, dropout=dropout if dropout > 0 else 0)
        elif rnn_type == 'GRU':
            self.decoder_rnn = nn.GRU(dec_dim_embed, dec_dim_state, dec_num_layers, batch_first=True,
                                      bidirectional=False, dropout=dropout if dropout > 0 else 0)
        else:  # Default to RNN
            self.decoder_rnn = nn.RNN(dec_dim_embed, dec_dim_state, dec_num_layers, batch_first=True,
                                      bidirectional=False, dropout=dropout if dropout > 0 else 0)

        if enc_dim_state != dec_dim_state:
            self.enc_dec_fc = nn.Linear(enc_dim_state, dec_dim_state)
        else:
            self.enc_dec_fc = nn.Identity()

        self.decoder_fc = nn.Linear(dec_dim_state, dec_dim_vocab)

    def forward(self, enc_x, max_len, bos_token_id, eos_token_id, dec_y_true=None, teacher_forcing=0, return_attn_weights=False):
        batch_size = enc_x.size(0)

        # Encode
        enc_x = self.encoder_embedding(enc_x)
        enc_outputs, hidden = self.encoder_rnn(enc_x)

        if isinstance(hidden, tuple):  # LSTM
            if self.encoder_num_directions == 2:
                hidden = (self._combine_directions_h(hidden[0]), self._combine_directions_h(hidden[1]))
                enc_outputs = self._combine_directions_o(enc_outputs)
            hidden = (self.enc_dec_fc(hidden[0]), self.enc_dec_fc(hidden[1]))
        else:  # GRU or RNN

            if self.encoder_num_directions == 2:
                hidden = self._combine_directions_h(hidden)
                enc_outputs = self._combine_directions_o(enc_outputs)
            hidden = self.enc_dec_fc(hidden)

        # Prepare the initial input to the decoder (BOS token)
        dec_input = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=enc_x.device)
        dec_input = self.decoder_embedding(dec_input)

        # Decode
        dec_y = torch.empty(0).to(dec_input.device.type)
        attention_weights = []
        for t in range(max_len):
            attn_weights = self._calculate_attention(hidden, enc_outputs)
            attention_weights.append(attn_weights)
            context = attn_weights.bmm(enc_outputs)

            dec_input_combined = torch.cat((dec_input.squeeze(1), context.squeeze(1)), 1)
            dec_input_combined = self.attn_combine(dec_input_combined).unsqueeze(1)

            output, hidden = self.decoder_rnn(dec_input_combined, hidden)
            output = self.decoder_fc(output.squeeze(1))
            dec_y = torch.cat((dec_y, output.unsqueeze(1)), dim=1)

            if dec_y_true is not None and torch.rand(1).item() < teacher_forcing:
                if t >= dec_y_true.size(1):
                    break
                dec_input = self.decoder_embedding(dec_y_true[:, t].unsqueeze(1))
            else:
                dec_input = self.decoder_embedding(output.argmax(1).unsqueeze(1))

            if torch.all(torch.any(dec_y.argmax(-1) == eos_token_id, dim=-1)):
                break

        attention_weights = torch.stack(attention_weights, dim=-1).squeeze(1) # shape [batch_size, src_timesteps, tgt_timesteps]
        if return_attn_weights:
            return dec_y, attention_weights
        else:
            return dec_y

    def _calculate_attention(self, hidden, enc_outputs):
        # hidden shape: (num_layers, batch_size, hidden_size)
        # enc_outputs shape: (batch_size, seq_len, hidden_size * num_directions)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden_expanded = hidden[-1].unsqueeze(2)
        attn_scores = torch.bmm(enc_outputs, hidden_expanded).squeeze(2)
        attn_weights = self.softmax(attn_scores)
        return attn_weights.unsqueeze(1)

    def _combine_directions_h(self, hidden):
        # hidden shape is (num_layers * num_directions, batch_size, hidden_size)
        # we need to combine the directions
        num_layers = hidden.size(0) // self.encoder_num_directions
        batch_size = hidden.size(1)
        hidden_size = hidden.size(2)

        # Combine the bidirectional hidden states
        combined = hidden.view(num_layers, self.encoder_num_directions, batch_size, hidden_size)
        combined = combined.sum(dim=1)
        return combined
    
    def _combine_directions_o(self, outputs):
        # outputs shape is (batch_size, seq_len, hidden_size * num_directions)
        # we need to combine the directions
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)
        hidden_size = outputs.size(2) // self.encoder_num_directions

        # Combine the bidirectional hidden states
        combined = outputs.view(batch_size, seq_len, hidden_size, self.encoder_num_directions)
        combined = combined.sum(dim=-1)
        return combined

    def pad_outputs(self, outputs, tgt_tokens, pad_token_id):
        if tgt_tokens.size(1) <= outputs.size(1):
            tgt_tokens = torch.cat((tgt_tokens, torch.full((tgt_tokens.size(0), outputs.size(1)-tgt_tokens.size(1)), pad_token_id).to(outputs.device.type)), dim=1)
        else:
            padded_one_hot = nn.functional.one_hot(torch.full((tgt_tokens.size(0),tgt_tokens.size(1)-outputs.size(1)), pad_token_id).to(outputs.device.type), num_classes=outputs.size(-1))
            outputs = torch.cat((outputs, padded_one_hot), dim=1)
        return outputs, tgt_tokens
    
class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim_state, dec_dim_state):
        super(BahdanauAttention, self).__init__()
        self.W_attn = nn.Linear(enc_dim_state, dec_dim_state)
        self.U_attn = nn.Linear(dec_dim_state, dec_dim_state)
        self.V_attn = nn.Linear(dec_dim_state, 1)

    def forward(self, dec_input, enc_outputs):
        # dec_input shape: (batch_size, 1, dec_dim_embed)
        # enc_outputs shape: (batch_size, seq_len, enc_dim_state)
        # print(dec_input.shape, enc_outputs.shape)
        query = self.W_attn(dec_input.squeeze(1))
        values = self.U_attn(enc_outputs.transpose(0,1).contiguous())

        # Expand dimensions for broadcasting
        query = query.unsqueeze(1)  # (batch_size, 1, dec_dim_state)
        values.transpose(0,1).contiguous()
        energy = torch.tanh(query + values.transpose(0,1).contiguous())  # (batch_size, seq_len, dec_dim_state)
        attention_scores = self.V_attn(energy).squeeze(-1)  # (batch_size, seq_len)

        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), enc_outputs)  # (batch_size, 1, enc_dim_state)
        # context_vector is the new tensor using attention that will be regarded as encoder_hidden to the decoder
        return context_vector, attention_weights


class VanillaSeq2SeqRNN(nn.Module):
    def __init__(self, enc_dim_vocab, enc_dim_embed, enc_dim_state, enc_num_layers,
                 dec_dim_vocab, dec_dim_embed, dec_dim_state, dec_num_layers,
                 bidirectional=True, dropout=0.1, rnn_type='RNN'):
        super(VanillaSeq2SeqRNN, self).__init__()

        # Encoder
        self.encoder_embedding = nn.Embedding(enc_dim_vocab, enc_dim_embed)
        if rnn_type == 'LSTM':
            self.encoder_rnn = nn.LSTM(enc_dim_embed, enc_dim_state, enc_num_layers, batch_first=True,
                                       bidirectional=bidirectional, dropout=dropout if dropout > 0 else 0)
            self.encoder_num_directions = 2 if bidirectional else 1
        elif rnn_type == 'GRU':
            self.encoder_rnn = nn.GRU(enc_dim_embed, enc_dim_state, enc_num_layers, batch_first=True,
                                      bidirectional=bidirectional, dropout=dropout if dropout > 0 else 0)
            self.encoder_num_directions = 2 if bidirectional else 1
        else:  # Default to RNN
            self.encoder_rnn = nn.RNN(enc_dim_embed, enc_dim_state, enc_num_layers, batch_first=True,
                                       bidirectional=bidirectional, dropout=dropout if dropout > 0 else 0)
            self.encoder_num_directions = 2 if bidirectional else 1

        # Decoder
        self.decoder_embedding = nn.Embedding(dec_dim_vocab, dec_dim_embed)
        if rnn_type == 'LSTM':
            self.decoder_rnn = nn.LSTM(dec_dim_embed, dec_dim_state, dec_num_layers, batch_first=True,
                                       bidirectional=False, dropout=dropout if dropout > 0 else 0)
        elif rnn_type == 'GRU':
            self.decoder_rnn = nn.GRU(dec_dim_embed, dec_dim_state, dec_num_layers, batch_first=True,
                                      bidirectional=False, dropout=dropout if dropout > 0 else 0)
        else:  # Default to RNN
            self.decoder_rnn = nn.RNN(dec_dim_embed, dec_dim_state, dec_num_layers, batch_first=True,
                                       bidirectional=False, dropout=dropout if dropout > 0 else 0)

        if enc_dim_state != dec_dim_state:
            self.enc_dec_fc = nn.Linear(enc_dim_state, dec_dim_state)
        else:
            self.enc_dec_fc = nn.Identity()
        
        self.decoder_fc = nn.Linear(dec_dim_state, dec_dim_vocab)

    def forward(self, enc_x, max_len, bos_token_id, eos_token_id, dec_y_true=None, teacher_forcing=0):
        batch_size = enc_x.size(0)

        # Encode
        enc_x = self.encoder_embedding(enc_x)
        _, hidden = self.encoder_rnn(enc_x)
        # print(_.shape, hidden.shape)

        if isinstance(hidden, tuple):
            if self.encoder_num_directions == 2:
                hidden = (self._combine_directions(hidden[0]), self._combine_directions(hidden[1]))
            hidden = self.enc_dec_fc(hidden[0]), self.enc_dec_fc(hidden[1])

        else:
            if self.encoder_num_directions == 2:
                hidden = self._combine_directions(hidden)
            hidden = self.enc_dec_fc(hidden)

        # Prepare the initial input to the decoder (BOS token)
        dec_input = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=enc_x.device)
        dec_input = self.decoder_embedding(dec_input)

        # Decode
        dec_y = torch.empty(0).to(dec_input.device.type)
        # dec_y = torch.zeros(batch_size, max_len, self.decoder_fc.out_features, device=enc_x.device)
        # FORWARD PASS FOR ALL TIMESTEPS UNTIL MAX_LEN OR EOS_TOKEN_ID
        for t in range(max_len):
            output, hidden = self.decoder_rnn(dec_input, hidden)
            output = self.decoder_fc(output.squeeze(1))
            dec_y = torch.cat((dec_y, output.unsqueeze(1)), dim=1)

            if dec_y_true is not None and torch.rand(1).item() < teacher_forcing:
                if t >= dec_y_true.size(1):
                    break
                dec_input = self.decoder_embedding(dec_y_true[:, t].unsqueeze(1))
            else:
                dec_input = self.decoder_embedding(output.argmax(1).unsqueeze(1))

            # if eos_token_id in output.argmax(1).tolist():
            #     break
            # Break if we have generated eos_token_id in each of our batches
            if torch.all(torch.any(dec_y.argmax(-1) == eos_token_id, dim=-1)):
                #"Breaking")
                break
        
        return dec_y
    
    def _combine_directions(self, hidden):
        # hidden shape is (num_layers * num_directions, batch_size, hidden_size)
        # we need to combine the directions
        num_layers = hidden.size(0) // self.encoder_num_directions
        batch_size = hidden.size(1)
        hidden_size = hidden.size(2)

        # Combine the bidirectional hidden states
        combined = hidden.view(num_layers, self.encoder_num_directions, batch_size, hidden_size)
        combined = combined.sum(dim=1)
        return combined
    
    def pad_outputs(self, outputs, tgt_tokens, pad_token_id):
        """
        As RNNs do not output a fixed sequence length, the predictions and targets may not be of the same shape
        which will cause an issue while computing the loss. Hence, we pad the lower dim tensor to match the higher.
        """
        # outputs.shape = (batch_size, timesteps, vocab_size)
        # tgt_tokens.shape = (batch_size, timesteps)

        if tgt_tokens.size(1) <= outputs.size(1):
            tgt_tokens = torch.cat((tgt_tokens, torch.full((tgt_tokens.size(0), outputs.size(1)-tgt_tokens.size(1)), pad_token_id).to(outputs.device.type)), dim=1)
        else: # outputs.size(1) < tgt_tokens.size(1)
            padded_one_hot = nn.functional.one_hot(torch.full((tgt_tokens.size(0),tgt_tokens.size(1)-outputs.size(1)), pad_token_id).to(outputs.device.type), num_classes=outputs.size(-1))
            outputs = torch.cat((outputs, padded_one_hot), dim=1)
            
        # print(outputs.shape, tgt_tokens.shape)
        return outputs, tgt_tokens

    

        



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__=="__main__":
    enc_dim_vocab, enc_dim_embed, enc_dim_state, enc_num_layers = 2048, 512, 256, 4
    dec_dim_vocab, dec_dim_embed, dec_dim_state, dec_num_layers = 2048, 512, 256, 4
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    batch_size, max_len = 64, 128

    seq2seq = VanillaSeq2SeqRNN(enc_dim_vocab, enc_dim_embed, enc_dim_state, enc_num_layers,
                         dec_dim_vocab, dec_dim_embed, dec_dim_state, dec_num_layers).to(DEVICE)
    
    enc_x = torch.randint(0, enc_dim_vocab, (batch_size, max_len)).to(DEVICE)
    dec_x = torch.full((batch_size,), bos_token_id).to(DEVICE)
    
    dec_y = seq2seq(enc_x, max_len, bos_token_id, eos_token_id)
    print(dec_y.shape)

    
    # enc_embedding = nn.Embedding(enc_dim_vocab, enc_dim_embed)
    # enc = RecurrentNeuralNetworkEncoder(enc_embedding.weight.to(DEVICE), enc_dim_state, enc_num_layers).to(DEVICE)
    # enc_h, enc_y = enc.forward(enc_x)
    # print(enc_h.shape, enc_y.shape)




    

