import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOEngine:
    """
    The Brain of the operation. Implements Proximal Policy Optimization (PPO).
    
    Hyperparameters Choice & Justification:
    ---------------------------------------
    clip_param (0.2): 
        Standard value. Prevents the new policy from diverging too wildly 
        from the old one in a single update.
        
    gamma (0.99):
        Discount factor. High value because in music, the "resolution" (cadence) 
        at the end is critical. We want earlier notes to care about the final cadence.
        
    gae_lambda (0.95):
        Smoothing factor for Advantage. Balances bias vs variance.
        
    entropy_coef (0.01):
        Crucial for creative tasks! Forces the model to keep trying diverse options
        instead of collapsing into a single "safe" melody too early (Mode Collapse).
        
    value_loss_coef (0.5):
        Standard balancing weight for the Critic's learning.
        
    max_grad_norm (0.5):
        Gradient clipping to prevent exploding gradients in the Transformer.
    """
    def __init__(self, model, optimizer, 
                 clip_param=0.2, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5):
        
        self.model = model
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

    def compute_gae(self, rewards, values, next_value, dones):
        """
        Generalized Advantage Estimation (GAE).
        Calculates 'Advantages': How much better was an action than expected?
        and 'Returns': The actual discounted sum of future rewards.
        """
        advantages = []
        returns = []
        gae = 0
        
        # Iterate backwards
        # values includes the values for the states in the episode
        # next_value is V(s_last+1) (usually 0 if done)
        
        # We align lists: validation needed on shapes
        # For simplicity in this script, we assume flattened lists or single batch processing
        
        values = values + [next_value] 
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            
            # Return = Advantage + Value (Target for Value Net)
            returns.insert(0, gae + values[step])
            
        return torch.tensor(advantages), torch.tensor(returns)

    def ppo_update(self, states, actions, old_log_probs, returns, advantages):
        """
        Performs the PPO update step.
        Normally this runs for K epochs on the collected batch.
        """
        # 1. Forward pass on CURRENT model
        # states: (Batch, Seq_Len)
        logits, values = self.model(states)
        
        # We need to extract the log_probs of the actions that were actually taken.
        # This requires careful gather operations if doing full sequence training.
        # Alternatively, PPO usually treats (State, Action) pairs as independent samples in the mini-batch.
        
        # Flatten for simplicity
        logits = logits.view(-1, logits.size(-1)) # (B*T, Vocab)
        actions = actions.view(-1)               # (B*T)
        values = values.view(-1)                 # (B*T)
        old_log_probs = old_log_probs.view(-1)
        returns = returns.view(-1)
        advantages = advantages.view(-1)
        
        # Calculate new log probs
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # 2. Ratio (New / Old)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 3. Surrogate Loss (Clipped)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 4. Value Loss
        # (Predicted Value - Real Return)^2
        value_loss = F.mse_loss(values, returns)
        
        # 5. Total Loss
        loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_coef * entropy)
        
        # 6. Optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item(), entropy.item()
