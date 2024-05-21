import torch
from torch import nn

from typing import Tuple, Optional
from torch import Tensor

class RecurrentNeuralNetworkCell(nn.Module):
    def __init__(self, dim_x:int, dim_h:int, dim_y:int):
        super().__init__()
        self.dim_x, self.dim_h, self.dim_y = dim_x, dim_h, dim_y

        self.U = nn.Parameter(torch.empty(dim_x, dim_h))
        self.W = nn.Parameter(torch.empty(dim_h, dim_h))
        self.b = nn.Parameter(torch.empty(dim_h))
        self.V = nn.Parameter(torch.empty(dim_h, dim_y))
        self.c = nn.Parameter(torch.empty(dim_y))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
        nn.init.constant_(self.b, 0.01)
        nn.init.constant_(self.c, 0.01)
    
    def x2h(self, x_curr:Tensor, h_prev:Tensor) -> Tensor:
        # x_curr.shape = torch.Size([batch_size, dim_x])
        # h_prev.shape = torch.Size([batch_size, dim_h])
        h_curr = torch.tanh(x_curr@self.U + h_prev@self.W + self.b)
        # h_curr.shape = torch.Size([batch_size, dim_h])
        return h_curr
    
    def h2y(self, h_curr:Tensor) -> Tensor:
        # h_curr.shape = torch.Size([batch_size, dim_h])
        y_curr = torch.softmax(h_curr@self.V + self.c,dim=-1)
        # y_curr.shape = torch.Size([batch_size, dim_y])
        return y_curr
    
    def forward(self, x_curr:Tensor, h_prev:Tensor) -> Tuple[Tensor, Tensor]:
        # x_curr.shape = torch.Size([batch_size, dim_x])
        # h_prev.shape = torch.Size([batch_size, dim_h])
        h_curr = self.x2h(x_curr, h_prev)
        # h_curr.shape = torch.Size([batch_size, dim_h])
        y_curr = self.h2y(h_curr)
        # y_curr.shape = torch.Size([batch_size, dim_y])
        return h_curr, y_curr
    
class RecurrentNeuralNetworkEncoderLayer(nn.Module):
    def __init__(self, dim_x:int, dim_h:int, dim_y:int):
        super().__init__()
        self.rnn_cell = RecurrentNeuralNetworkCell(dim_x, dim_h, dim_y)
        self.h_prev = nn.Parameter(torch.empty(self.rnn_cell.dim_h))
        nn.init.constant_(self.h_prev, 0.01)
    
    def forward(self, x:Tensor, h_prev:Tensor) -> Tuple[Tensor, Tensor]:
        h = torch.empty(0).to(x.device.type)
        y = torch.empty(0).to(x.device.type)

        if h_prev is None:
            h_prev = self.h_prev.repeat(x.size(0), 1).to(x.device.type) # Repeat the initial state vector across the batch size
        elif h_prev.dim() == 1:
            assert h_prev.size(0) == (self.rnn_cell.dim_h), "Shape of state vector is not as expected"
            h_prev = self.h_prev.repeat(x.size(0), 1).to(x.device.type) # Repeat the given state vector across the batch size
        else:
            assert h_prev.shape == torch.Size([x.size(0), self.rnn_cell.dim_h]), "Shape of state vector is not as expected"
        
        # FORWARD PASS FOR ALL TIMESTEPS
        for step in range(x.size(1)):
            x_curr = x[:,step,:]
            h_curr, y_curr = self.rnn_cell(x_curr, h_prev)
            h = torch.cat((h, h_curr.unsqueeze(1)), dim=1)
            y = torch.cat((y, y_curr.unsqueeze(1)), dim=1)
            h_prev = h_curr
        # h.shape = torch.Size([batch_size, timesteps, dim_state])
        # y.shape = torch.Size([batch_size, timesteps, dim_vocab])
        return h, y
    
class RecurrentNeuralNetworkEncoder(nn.Module):
    def __init__(self, embedding_matrix:Tensor, dim_state:int, num_layers:int):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        dim_vocab, dim_embed = embedding_matrix.shape
        
        if num_layers == 1:
            self.layers = nn.ModuleList([RecurrentNeuralNetworkEncoderLayer(dim_embed, dim_state, dim_vocab)])
        elif num_layers == 2:
            self.layers = nn.ModuleList([RecurrentNeuralNetworkEncoderLayer(dim_embed, dim_state, dim_state)] +
                                         [RecurrentNeuralNetworkEncoderLayer(dim_state, dim_state, dim_vocab)])
        else:                                
            self.layers = nn.ModuleList([RecurrentNeuralNetworkEncoderLayer(dim_embed, dim_state, dim_state)] +
                                    [RecurrentNeuralNetworkEncoderLayer(dim_state, dim_state, dim_state) for _ in range(num_layers-2)] +
                                    [RecurrentNeuralNetworkEncoderLayer(dim_state, dim_state, dim_vocab)])
            
        self.num_layers = num_layers
    
    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if x.dim() == 1: # if x.shape == torch.Size([batch_size])
            x = x.unsqueeze(1) # timesteps = 1
        # x.shape = torch.Size([batch_size, timesteps])
        x = nn.functional.embedding(x, self.embedding_matrix)
        # x.shape = torch.Size([batch_size, timesteps, dim_embed])
        
        
        for i,layer in enumerate(self.layers):
            # print(layer.rnn_cell.dim_x, layer.rnn_cell.dim_h, layer.rnn_cell.dim_y)
            h, y = layer(x, h_prev)
            # print(f"Layer ({i}): {h.shape} | {y.shape}")
            # h.shape = torch.Size([batch_size, timesteps, dim_state])
            # y.shape = torch.Size([batch_size, timesteps, dim_state/dim_vocab])
            x = y
        return h, y

class RecurrentNeuralNetworkDecoderLayer(nn.Module):
    def __init__(self, dim_embed:int, dim_state:int, dim_vocab:int, num_layers:int):
        super().__init__()
        self.dim_embed, self.dim_state, self.dim_vocab = dim_embed, dim_state, dim_vocab
        if num_layers == 1:
            self.layers = nn.ModuleList([RecurrentNeuralNetworkCell(dim_embed, dim_state, dim_vocab)])
        elif num_layers == 2:
            self.layers = nn.ModuleList([RecurrentNeuralNetworkCell(dim_embed, dim_state, dim_state)] +
                                         [RecurrentNeuralNetworkCell(dim_state, dim_state, dim_vocab)])
        else:                                
            self.layers = nn.ModuleList([RecurrentNeuralNetworkCell(dim_embed, dim_state, dim_state)] +
                                    [RecurrentNeuralNetworkCell(dim_state, dim_state, dim_state) for _ in range(num_layers-2)] +
                                    [RecurrentNeuralNetworkCell(dim_state, dim_state, dim_vocab)])
    
    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # x.shape = torch.Size([batch_size, dim_embed])
        h_layers = torch.empty(0).to(x.device.type)
        
        for i,layer in enumerate(self.layers):
            h, y = layer(x, h_prev[i,:,:])
            # h.shape = torch.Size([batch_size, dim_state])
            # y.shape = torch.Size([batch_size, dim_state/dim_vocab])
            x = torch.tanh(y)
            h_layers = torch.cat((h_layers, h.unsqueeze(0)), dim=0)
        
        # h_layers.shape = torch.Size([num_layers, batch_size, dim_state])
        return h_layers, y


class RecurrentNeuralNetworkDecoder(nn.Module):
    def __init__(self, embedding_matrix:Tensor, dim_state:int, num_layers:int):
        super().__init__()
        self.embedding_matrix = embedding_matrix
        dim_vocab, dim_embed = embedding_matrix.shape
        
        self.decoder = RecurrentNeuralNetworkDecoderLayer(dim_embed, dim_state, dim_vocab, num_layers)
        
        self.num_layers = num_layers

        self.h_prev = nn.Parameter(torch.empty(num_layers, dim_state))
        nn.init.xavier_uniform_(self.h_prev)

    def forward(self, x:Tensor, max_len:int, eos_token_id:int, h_prev:Optional[Tensor]=None, y_true:Optional[Tensor]=None, teacher_forcing:Optional[int]=0) -> Tuple[Tensor, Tensor]:
        assert x.dim() == 1, "todo: Should work even if multiple beginning time steps are passed"
        if x.dim() == 1: # if x.shape == torch.Size([batch_size])
            x = x.unsqueeze(1) # timesteps = 1
        # x.shape = torch.Size([batch_size, timesteps])
        x = nn.functional.embedding(x, self.embedding_matrix)
        # x.shape = torch.Size([batch_size, timesteps, dim_embed])

        h = torch.empty(0).to(x.device.type)
        y = torch.empty(0).to(x.device.type)
        
        if h_prev is None:
            h_prev = self.h_prev.repeat(self.num_layers, x.size(0), 1).to(x.device.type) # Repeat the initial state vector across the batch size for all layers
        elif h_prev.dim() == 1:
            assert h_prev.size(0) == x.size(0), "Shape of state vector is not as expected"
            h_prev = self.h_prev.unsqueeze(0).repeat(self.num_layers, x.size(0), 1).to(x.device.type) # Repeat the given state vector across the batch size for all layers
        else:
            assert h_prev.shape == torch.Size([x.size(0), self.decoder.dim_state]), "Shape of state vector is not as expected"
            h_prev = h_prev.repeat(self.num_layers, 1, 1).to(x.device.type)
            # print(h_prev.shape)

            
        
        # print(y_true.shape) # y.shape = torch.Size([batch_size, timesteps])
        if teacher_forcing!=0:
            y_true = nn.functional.embedding(y_true, self.embedding_matrix) # y.shape = torch.Size([batch_size, timesteps, dim_vocab])
        # FORWARD PASS FOR ALL TIMESTEPS UNTIL MAX_LEN OR EOS_TOKEN_ID
        n=0
        for step in range(max_len):
            if step == 0 or teacher_forcing < torch.rand((1)).item():
                x_curr = x[:,step,:]
            else:
                n+=1
                if step >= y_true.size(1):
                    break
                x_curr = y_true[:,step,:]
            
            h_layers, y_curr = self.decoder(x_curr, h_prev)
            h = torch.cat((h, h_layers[-1,:,:].unsqueeze(1)), dim=1)
            y = torch.cat((y, y_curr.unsqueeze(1)), dim=1)
            h_prev = h_layers
            
            x_next = torch.argmax(y_curr, dim=-1)
            # x_next.shape = torch.Size([batch_size])
            x_next = nn.functional.embedding(x_next, self.embedding_matrix)
            # x_next.shape = torch.Size([batch_size, dim_embed])
            # print(x_next.shape)
            x = torch.cat((x, x_next.unsqueeze(1)), dim=1)
            
            # Break if we have generated eos_token_id in each of our batches
            x_temp = torch.argmax(x, dim=-1)
            if torch.all(torch.any(x_temp == eos_token_id, dim=-1)):
                print("Breaking")
                break
        # h.shape = torch.Size([batch_size, timesteps, dim_state])
        # y.shape = torch.Size([batch_size, timesteps, dim_vocab])
        # print(f"Teacher forced {n}/{step} times")
        return h, y
    

    

class VanillaSeq2SeqRNN(nn.Module):
    def __init__(self, 
                 enc_dim_vocab:int, enc_dim_embed:int, enc_dim_state:int, enc_num_layers,
                 dec_dim_vocab:int, dec_dim_embed:int, dec_dim_state:int, dec_num_layers,
                 bidirectional=False, dropout=0, rnn_type='RNN'):
        super().__init__()
        self.enc_embedding = nn.Embedding(enc_dim_vocab, enc_dim_embed)
        self.dec_embedding = nn.Embedding(dec_dim_vocab, dec_dim_embed)
        self.encoder = RecurrentNeuralNetworkEncoder(self.enc_embedding.weight, enc_dim_state, enc_num_layers)
        if enc_dim_state != dec_dim_state:
            self.linear = nn.Linear(enc_dim_state, dec_dim_state)
        else:
            self.linear = nn.Identity()
        self.decoder = RecurrentNeuralNetworkDecoder(self.dec_embedding.weight, dec_dim_state, dec_num_layers)
    
    def forward(self, enc_x:Tensor, max_len:int, bos_token_id:int, eos_token_id:int, dec_y_true:Optional[Tensor]=None, teacher_forcing:Optional[int]=0) -> Tensor:
        # print(enc_x.shape)
        enc_h, enc_y = self.encoder(enc_x)
        # print(enc_h.shape, enc_y.shape)
        dec_h_prev = self.linear(enc_h[:,-1,:])
        # print(dec_h_prev.shape)
        dec_x = torch.full((enc_x.size(0),), bos_token_id).to(DEVICE)
        # print(dec_x.shape)

        dec_h, dec_y = self.decoder(dec_x, max_len, eos_token_id, dec_h_prev, dec_y_true, teacher_forcing)
        # print(dec_h.shape, dec_y.shape)
        
        # return enc_h, enc_y, dec_h, dec_y
        return dec_y
    
    def pad_outputs(self, outputs, tgt_tokens, pad_token_id):
        """
        As RNNs do not output a fixed sequence length, the predictions and targets may not be of the same shape
        which will cause an issue while computing the loss. Hence, we pad the lower dim tensor to match the higher.
        """
        # outputs.shape = (batch_size, timesteps, vocab_size)
        # tgt_tokens.shape = (batch_size, timesteps)

        # if outputs.size(1) >= tgt_tokens.size(1): # If outputs
        #     # tgt_tokens = torch.cat((tgt_tokens, torch.full((tgt_tokens.size(0), outputs.size(1)-tgt_tokens.size(1)), pad_token_id).to("cuda")), dim=-1)
        #     outputs = outputs[:,:tgt_tokens.size(1),:].contiguous()
        # else:
        #     tgt_tokens = tgt_tokens[:,outputs.size(1),:].contiguous()

        if tgt_tokens.size(1) <= outputs.size(1):
            tgt_tokens = torch.cat((tgt_tokens, torch.full((tgt_tokens.size(0), outputs.size(1)-tgt_tokens.size(1)), pad_token_id).to(outputs.device.type)), dim=1)
        else: # outputs.size(1) < tgt_tokens.size(1)
            padded_one_hot = nn.functional.one_hot(torch.full((tgt_tokens.size(0),tgt_tokens.size(1)-outputs.size(1)), pad_token_id).to(outputs.device.type), num_classes=outputs.size(-1))
            outputs = torch.cat((outputs, padded_one_hot), dim=1)
            
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




    

