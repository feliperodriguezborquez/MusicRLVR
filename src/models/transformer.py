import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CounterpointTransformer(nn.Module):
    """
    A simple Decoder-only Transformer for autoregressive music generation.
    Or Encoder-Decoder?
    Since we have CF (Source) -> CP (Target), Encoder-Decoder is natural.
    But GPT-style Decoder-only (with <CF>...<CP> prompt) is also fine and simpler for RL.
    We'll use Decoder-only for MVP (GPT style).
    """
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return self.fc_out(output)

    def generate(self, start_tokens, max_new_tokens=16, temperature=1.0):
        """
        Greedy/Sampled generation loop (MVP).
        """
        self.eval()
        curr_seq = start_tokens # (1, seq_len)
        generated = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(curr_seq)
                last_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(last_logits, dim=-1)
                
                next_token = torch.multinomial(probs, 1)
                curr_seq = torch.cat([curr_seq, next_token], dim=1)
                generated.append(next_token.item())
                
                # Check for EOS if we had one
                
        return generated
