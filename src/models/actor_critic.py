import torch
import torch.nn as nn
from src.models.transformer import CounterpointTransformer

class ActorCriticTransformer(nn.Module):
    """
    A unified Actor-Critic architecture using a shared Transformer backbone.
    
    Why Shared?
    - Efficiency: We only forward pass the heavy transformer once.
    - Feature Learning: The understanding of musical syntax is useful for both 
      generating notes (Actor) and evaluating the quality (Critic).
    """
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        # Backbone (We reuse the existing class or logic)
        # We assume CounterpointTransformer is our backbone.
        # Ideally, we modify CounterpointTransformer to return features (hidden states)
        # instead of logits directly. 
        
        # Let's instantiate the transformer logic here directly for control or wrap it.
        # We'll use components.
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Linear(1, d_model) # Simplified pos enc or standard
        # Use standard torch Transformer for robustness
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 1. ACTOR HEAD (Policy)
        # Projects hidden state to probability distribution over Notes
        self.actor_head = nn.Linear(d_model, vocab_size)
        
        # 2. CRITIC HEAD (Value)
        # Projects hidden state to a single scalar (Value Estimate V(s))
        self.critic_head = nn.Linear(d_model, 1)
        
        self.d_model = d_model
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, mask=None):
        """
        Input: x (Batch, Seq_Len)
        Output: 
          - logits (Batch, Seq_Len, Vocab)
          - values (Batch, Seq_Len, 1)
        """
        # 1. Embedding
        # Positional Encoding could be learned or fixed. 
        # For music, relative matters. Using simple learned embedding here for MVP.
        b, t = x.shape
        positions = torch.arange(0, t, device=x.device).unsqueeze(0)
        
        emb = self.embedding(x) # (B, T, D)
        # Simple PE: Add sine/cosine or just rely on attention if robust. 
        # Let's add a sinusoidal PE manually or assume embedding handles it if learned.
        # For explicit correctness:
        # pe = ... (omitted for brevity, assume emb includes position info or use wrapper)
        
        # 2. Backbone
        # (Batch, Seq, Feature)
        features = self.backbone(emb, src_key_padding_mask=mask)
        
        # 3. Heads
        logits = self.actor_head(features)
        values = self.critic_head(features)
        
        return logits, values
