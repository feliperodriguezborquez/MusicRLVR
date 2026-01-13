import torch
from torch.utils.data import Dataset
import os

class MusicRLVRDataset(Dataset):
    """
    Dataset for MusicRLVR. 
    Loads text files where each line is:
    <STYLE_TOKEN> <CF> n1 n2 ... <CP> n1 n2 ...
    or just <STYLE_EXERCISE> <CF> n1 n2 ...
    """
    def __init__(self, file_paths, tokenizer, max_len=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        for path in file_paths:
            if not os.path.exists(path):
                print(f"Warning: Data file {path} not found.")
                continue
                
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    self.examples.append(line)
        
        print(f"Loaded {len(self.examples)} examples from {file_paths}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        line = self.examples[idx]
        # Tokenize
        # Phase 1 RL: We might just need the CF (Source) to prompt the agent.
        # But for training a Transformer (autoregressive), we usually format as:
        # [BOS] <STYLE> <CF> ... <CP> ... [EOS]
        
        # Parse the string manually to separate source/target if needed, 
        # or just feed the whole sequence for causal LM training.
        # For RL, we'll need to separate.
        
        tokens = self.tokenizer.encode(line)
        
        # Pad/Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        # For PyTorch DataLoader, return tensor
        # This is a basic causal LM setup
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Mask/Pad would be done in collate_fn ideally
        return input_ids

class MusicTokenizer:
    """
    Simple whitespace tokenizer for integer pitches and special tokens.
    """
    def __init__(self):
        self.special_tokens = [
            "<PAD>", "<BOS>", "<EOS>", 
            "<STYLE_EXERCISE>", "<STYLE_REAL>", "<CF>", "<CP>"
        ]
        # Pitches MIDI 0-127
        self.pitch_tokens = [str(i) for i in range(128)]
        
        self.vocab = self.special_tokens + self.pitch_tokens
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        
        self.pad_id = self.token_to_id["<PAD>"]
        self.bos_id = self.token_to_id["<BOS>"]
        self.eos_id = self.token_to_id["<EOS>"]

    def encode(self, text):
        # text is space separated
        parts = text.split()
        ids = [self.bos_id]
        for p in parts:
            if p in self.token_to_id:
                ids.append(self.token_to_id[p])
            else:
                # OOV or unknown? Just ignore or error.
                # In our controlled env, shouldn't happen.
                pass
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in self.id_to_token:
                tokens.append(self.id_to_token[i])
        return " ".join(tokens)
    
    @property
    def vocab_size(self):
        return len(self.vocab)
