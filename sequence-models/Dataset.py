import os

from torch.utils.data import Dataset, DataLoader
import torch

class AksharantarTokenizer:
    def __init__(self, words:list, bos_token=None, eos_token=None, pad_token=None):
        vocab = []
        for word in words:
            for char in word:
                if char not in vocab:
                    vocab.append(char)

        if eos_token is not None:
            if eos_token in vocab:
                print(f"Warning: eos_token '{eos_token}', is already a character in the dataset. Using it as eos_token will overwrite its behaviour.")
            else:
                vocab = [eos_token] + vocab
        else:
            self.eos_token_id = None
        self.eos_token = eos_token
        
        if bos_token is not None:
            if bos_token in vocab:
                print(f"Warning: bos_token '{bos_token}', is already a character in the dataset. Using it as bos_token will overwrite its behaviour.")
            else:
                vocab = [bos_token] + vocab
        else:
            self.bos_token_id = None
        self.bos_token = bos_token
        
        if pad_token is not None:
            if pad_token in vocab:
                print(f"Warning: pad_token '{pad_token}', is already a character in the dataset. Using it as pad_token will overwrite its behaviour.")
            else:
                vocab = [pad_token] + vocab
        else:
            self.pad_token_id = None
        self.pad_token = pad_token

        self.eos_token_id = vocab.index(eos_token)
        self.bos_token_id = vocab.index(bos_token)
        self.pad_token_id = vocab.index(pad_token)
            
        self.vocab = vocab
        # print(len(self.vocab))
        # print(self.vocab)
    
    def __len__(self):
        return len(self.vocab)

    def encode(self, text:str):
        return [self.vocab.index(char) for char in text]
    
    def decode(self, indices:list):
        decoded_tokens = []
        for index in indices:
            if index == self.pad_token_id or index == self.eos_token_id:
               break
            decoded_tokens.append(self.vocab[index])
        return ''.join(decoded_tokens)
    
    def decode_sequences(self, list_of_lists:list):
        decoded_sequences = []
        for token_ids in list_of_lists:
            decoded_sequence = self.decode(token_ids)
            decoded_sequences.append(decoded_sequence)
        return decoded_sequences


        



class Aksharantar(Dataset):
    def __init__(self, data_dir:str, lang:str, split:str):
        bitext = {"src":[], "tgt":[]}

        with open(os.path.join(data_dir,lang,f"{lang}_{split}.csv")) as f:
            line = f.readline()
            while line:
                src, tgt = line.split(",")
                src, tgt = src.strip(), tgt.strip()
                assert " " not in src and " " not in tgt, "Sanity check no spaces in text"
                bitext["src"].append(src)
                bitext["tgt"].append(tgt)
                line = f.readline()
                    
        self.bitext = bitext
    
    def tokenize(self, src_tokenizer, tgt_tokenizer):
        tokenized_bitext = dict()
        tokenized_bitext["src"] = [src_tokenizer.encode(x) for x in self.bitext["src"]]
        tokenized_bitext["tgt"] = [tgt_tokenizer.encode(x)+[tgt_tokenizer.eos_token_id] for x in self.bitext["tgt"]]
        # tokenized_bitext["tgt"] = []
        # for x in self.bitext["tgt"]:
        #   # tokenized_bitext["tgt"].append(tgt_tokenizer.encode(x)+[tgt_tokenizer.eos_token_id])

        special_tokens = dict()
        
        special_tokens["src"] = {"pad": {"token":src_tokenizer.pad_token, "id":src_tokenizer.pad_token_id}, 
                                 "bos": {"token":src_tokenizer.bos_token, "id":src_tokenizer.bos_token_id}, 
                                 "eos": {"token":src_tokenizer.eos_token, "id":src_tokenizer.eos_token_id}}
        
        special_tokens["tgt"] = {"pad": {"token":tgt_tokenizer.pad_token, "id":tgt_tokenizer.pad_token_id}, 
                                 "bos": {"token":tgt_tokenizer.bos_token, "id":tgt_tokenizer.bos_token_id}, 
                                 "eos": {"token":tgt_tokenizer.eos_token, "id":tgt_tokenizer.eos_token_id}}
        
        self.special_tokens = special_tokens
        self.tokenized_bitext = tokenized_bitext

    def __getitem__(self, index):
        src = torch.LongTensor(self.tokenized_bitext["src"][index])
        tgt = torch.LongTensor(self.tokenized_bitext["tgt"][index])
        return src, tgt
    
    def __len__(self):
        return len(self.bitext["src"])

    def collate_fn(self, batch):
        src_list, tgt_list = zip(*batch)
        batch_src = torch.nn.utils.rnn.pad_sequence(src_list, padding_value=self.special_tokens["src"]["pad"]["id"], batch_first=True)
        batch_tgt = torch.nn.utils.rnn.pad_sequence(tgt_list, padding_value=self.special_tokens["tgt"]["pad"]["id"], batch_first=True)
        return batch_src, batch_tgt

    
        


        




if __name__=="__main__":
    data_dir = "../aksharantar_sampled/"
    lang = "hin"
    
    train_set = Aksharantar(data_dir, lang, "train")
    valid_set = Aksharantar(data_dir, lang, "valid")
    test_set = Aksharantar(data_dir, lang, "test")
    
    src_tokenizer = AksharantarTokenizer(train_set.bitext["src"]+valid_set.bitext["src"]+test_set.bitext["src"], bos_token="+", eos_token=".", pad_token=",")
    tgt_tokenizer = AksharantarTokenizer(train_set.bitext["tgt"]+valid_set.bitext["tgt"]+test_set.bitext["tgt"], bos_token="+", eos_token=".", pad_token=",")
    
    train_set.tokenize(src_tokenizer, tgt_tokenizer)
    valid_set.tokenize(src_tokenizer, tgt_tokenizer)
    test_set.tokenize(src_tokenizer, tgt_tokenizer)

    print(src_tokenizer.decode(train_set.tokenized_bitext["src"][0]))

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size, collate_fn=valid_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size, collate_fn=test_set.collate_fn)
    
    batch = next(iter(train_loader))
    src, tgt = batch
    print(src.shape, tgt.shape)

    

                




