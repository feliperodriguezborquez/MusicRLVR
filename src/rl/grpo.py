import torch
import torch.nn.functional as F


def masked_kl_penalty(log_probs, ref_log_probs, mask, beta):
    """Compute GRPO KL only at active positions.

    Exclude padding before subtraction and exp, so finite differences at
    inactive positions cannot overflow and poison backward through 0 * inf.
    """
    active = mask != 0
    current = torch.where(active, log_probs, 0.0)
    reference = torch.where(active, ref_log_probs, 0.0)
    log_ratio = current - reference
    return beta * (torch.exp(log_ratio) - log_ratio - 1)


class GRPOEngine:
    """
    Group Relative Policy Optimization (GRPO).
    The technique behind reasoning models like DeepSeek-R1.
    
    Key Features:
    - No Critic (Value Network) required.
    - Baseline = Mean reward of the group samples.
    - Advantages are normalized within the group.
    - Explicit KL penalty against Reference Model.
    """
    def __init__(self, model, ref_model, optimizer, 
                 beta=0.04,          # KL Penalty coefficient (DeepSeek uses 0.001 - 0.04 range)
                 clip_param=0.2):
        self.model = model
        self.ref_model = ref_model # Frozen reference
        self.optimizer = optimizer
        self.beta = beta
        self.clip_param = clip_param

    def compute_loss(self, inputs, generated_ids, rewards, mask_generated):
        """
        inputs: context (CF)
        generated_ids: Full sequences [CF + CP] (Batch*G, SeqLen)
        rewards: Scalar reward for each sequence (Batch*G)
        mask_generated: Boolean mask indicating which tokens are generated (Target) vs prompt.
        """
        # 1. Forward Pass (Current Policy)
        logits = self.model(generated_ids)
        
        # Shift logits for casual LM loss
        # logits[t] predicts id[t+1]
        # We are interested in log_prob of generated tokens
        
        # log_probs: (B*G, L)
        # We gather the log_probs of the actual tokens
        all_log_probs = logits.log_softmax(dim=-1)
        token_log_probs = torch.gather(all_log_probs, -1, generated_ids.unsqueeze(-1)).squeeze(-1)
        
        # Shift: token_log_probs matches generated_ids.
        # But logits at t correspond to generated_ids[t+1].
        # Let's align carefully.
        # logits[:, :-1] predicts generated_ids[:, 1:]
        
        logits_aligned = logits[:, :-1, :]
        ids_aligned = generated_ids[:, 1:]
        mask_aligned = mask_generated[:, 1:]
        
        # Re-gather
        log_probs_dist = logits_aligned.log_softmax(dim=-1)
        log_probs = torch.gather(log_probs_dist, -1, ids_aligned.unsqueeze(-1)).squeeze(-1)
        
        # 2. Forward Pass (Ref Policy) - No Grad
        with torch.no_grad():
            ref_logits = self.ref_model(generated_ids)
            ref_logits_aligned = ref_logits[:, :-1, :]
            ref_dist = ref_logits_aligned.log_softmax(dim=-1)
            ref_log_probs = torch.gather(ref_dist, -1, ids_aligned.unsqueeze(-1)).squeeze(-1)

        # 3. Compute KL (approx) per token
        # kl = exp(ref - curr) - (ref - curr) - 1  (Schulman approx)
        # Or simple: log(pi) - log(ref)
        # DeepSeek R1 Paper Formula:
        # Loss = E [ min(ratio*A, clip*A) - beta * D_KL(pi || ref) ]
        # Where D_KL is approximated as (log_pi - log_ref)
        
        # Let's use the explicit per-token KL for the penalty term
        # ratio_kl = log_probs - ref_log_probs
        # kl_penalty = self.beta * ratio_kl 
        # Actually in GRPO formula, the KL is subtracted from the advantage objective
        # Objective = Gain - Penalty
        # Loss = -Objective
        
        ratio_log = log_probs - ref_log_probs
        # KL term is usually exp(ratio_log) - ratio_log - 1 for consistency
        # But simple version: beta * ratio_log (since ratio_log is log(pi/ref))
        kl_penalty = self.beta * (torch.exp(ratio_log) - ratio_log - 1) 
        
        # 4. Computed Advantages (Group Relative)
        # rewards shape: (Batch * G)
        # We need to reshape to (Batch, G) to compute mean/std
        G = rewards.size(0) // inputs.size(0) # Assuming uniform group size
        B = inputs.size(0)
        
        rewards_view = rewards.view(B, G)
        mean_r = rewards_view.mean(dim=1, keepdim=True)
        std_r = rewards_view.std(dim=1, keepdim=True) + 1e-8
        
        adv = (rewards_view - mean_r) / std_r # (B, G)
        adv = adv.view(-1) # Flatten back to (B*G)
        
        # Expand advantage to token level?
        # In PPO, Advantage is per token (GAE).
        # In GRPO (and many LLM RL), we often use outcome reward for ALL result tokens.
        # So Adv[token_t] = Adv[sequence]
        
        adv_expanded = adv.unsqueeze(1).repeat(1, log_probs.size(1)) # (B*G, L)
        
        # 5. Ratio (PPO style importance sampling if we do multiple epochs)
        # For single batch update (Online), ratio is 1.0 (log_probs == old_log_probs).
        # But if we do multiple robust steps:
        # We would need 'old_log_probs' detached.
        # For simplicity Step 1: ratio = 1.0. derivative flows through log_probs.
        # But wait, GRPO usually iterates.
        # Let's assume we detached 'old_log_probs' in checking logic. 
        # Here we just implement the loss against "old" (which is current for first step).
        
        # Current implementation: Online 1-step (Ratio=1).
        # To make it proper PPO, we need 'old_log_probs' passed in.
        # Assuming `log_probs` is the active one, and we create `old_log_probs` before optimization loop.
        
        # Simplified for this class:
        # Loss = - (Adv * log_probs) + KL
        # (This is Vanilla PG with Baseline).
        # With Clipping:
        # ratio = exp(log_probs - old_log_probs)
        # ...
        
        pass

    def ppo_loss(self, log_probs, old_log_probs, advantages, kl_penalty, mask):
        """
        Calculates the GRPO/PPO clipped loss.
        """
        active = mask != 0
        current = torch.where(active, log_probs, 0.0)
        old = torch.where(active, old_log_probs, 0.0)
        ratio = torch.exp(current - old)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        
        # Objective: Maximize (Min(surr1, surr2) - KL)
        # Loss: Minimize -(...)
        
        objective = torch.min(surr1, surr2) - kl_penalty
        
        # Mask out non-generated tokens (padding or prompt)
        valid_objective = objective * mask
        
        # Sum over sequence, mean over batch
        loss = -valid_objective.sum() / mask.sum()
        
        return loss
