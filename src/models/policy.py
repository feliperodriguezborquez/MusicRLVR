import torch
import torch.nn as nn
from src.models.transformer import CounterpointTransformer

class PolicyTransformer(nn.Module):
    """
    Pure Actor Model (Policy Network).
    Corrected for Autoregressive Causal Masking.
    """
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, d_model)) 
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def _generate_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, mask=None):
        """
        x: (Batch, Seq_Len)
        """
        b, t = x.shape
        
        # Create Causal Mask (prevent peeking forward)
        # Device matching handled by register_buffer if persisted, but dynamic is fine for variable len
        causal_mask = self._generate_causal_mask(t).to(x.device)
        
        # Combine with padding mask if provided
        # Note: PyTorch Transformer expects:
        # src_mask: (T, T) for causality
        # src_key_padding_mask: (B, T) for padding
        
        emb = self.embedding(x) + self.pos_emb[:, :t, :]
        
        # Apply mask
        features = self.backbone(emb, mask=causal_mask, src_key_padding_mask=mask)
        logits = self.head(features)
        
        return logits
    
    def generate_group(self, input_ids, max_new_tokens, group_size=4, temperature=1.0):
        """
        Generates G samples.
        """
        batch_size = input_ids.size(0)
        curr_seq = input_ids.repeat_interleave(group_size, dim=0)
        
        self.eval()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Important: Forward pass needs causal mask even during generation?
                # Actually during generation we usually pass the full sequence so far.
                # The mask ensures we treat it autoregressively.
                
                logits = self.forward(curr_seq)
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                curr_seq = torch.cat([curr_seq, next_token], dim=1)
                
        return curr_seq
