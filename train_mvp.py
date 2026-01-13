import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import tqdm

from src.data.dataset import MusicRLVRDataset, MusicTokenizer
from src.models.policy import PolicyTransformer
from src.rl.grpo import GRPOEngine
from src.rl.rewards import CounterpointRules
from src.utils.reproducibility import seed_everything

CONFIG = {
    # GRPO Specific
    "group_size": 4,        
    "beta": 0.01,           
    
    # Curriculum
    "phases": {
        1: {"name": "Strict Rules", "dataset": "cantus_firmus_dataset.txt"},
        2: {"name": "Bach Style",   "dataset": "bach_soprano_bass.txt"},
    },
    
    "d_model": 128,
    "n_layers": 2,
    "n_heads": 4,
    "lr": 1e-5, 
    "total_epochs": 100,
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs"
}

class MusicGRPOTrainer:
    def __init__(self):
        seed_everything(42) # Reproducibility
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = MusicTokenizer()
        self.writer = SummaryWriter(log_dir=CONFIG["log_dir"])
        
        # 1. Active Policy
        print(f"Initializing Policy Model on {self.device}...")
        self.model = PolicyTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=CONFIG["d_model"],
            num_layers=CONFIG["n_layers"], 
            nhead=CONFIG["n_heads"]
        ).to(self.device)
        
        # 2. Reference Policy (Frozen)
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
            
        self.optimizer = optim.AdamW(self.model.parameters(), lr=CONFIG["lr"])
        
        # 3. GRPO Engine
        self.grpo = GRPOEngine(self.model, self.ref_model, self.optimizer, beta=CONFIG["beta"])
        
        # 4. Reward Engine
        self.examiner = CounterpointRules()
        
        os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
        
    def train_step(self, batch_cf_ids):
        """
        One GRPO optimization step (Batch -> Generate -> Update)
        """
        G = CONFIG["group_size"]
        B = batch_cf_ids.size(0)
        
        generated_seqs = self.model.generate_group(
            batch_cf_ids, 
            max_new_tokens=16, 
            group_size=G
        )
        
        prompt_len = batch_cf_ids.size(1)
        full_lists = generated_seqs.tolist()
        rewards_list = []
        
        for seq in full_lists:
            # Safely slice lists
            cf_part = seq[:prompt_len]
            cp_part = seq[prompt_len:]
            
            # Simple reward call (CPU)
            r = self.examiner.calculate_reward(cf_part, cp_part)
            rewards_list.append(r)
            
        rewards_tensor = torch.tensor(rewards_list).to(self.device)
        avg_reward = rewards_tensor.mean().item()
        
        # GRPO Update logic...
        # ... (Same as before, relying on model forward with causal mask capability) ...
        # Ensure we pass the mask to model if training logic calls it.
        # model.forward() inside GRPO usually handles full sequence.
        # But we need to ensure the mask is created. PolicyTransformer.forward creates it.
        
        # Re-calc logits for update
        logits = self.model(generated_seqs) # Forward will generate mask
        
        logits = logits[:, :-1, :]
        labels = generated_seqs[:, 1:]
        
        dist = logits.log_softmax(dim=-1)
        old_log_probs = torch.gather(dist, -1, labels.unsqueeze(-1)).squeeze(-1)
        
        mask = torch.zeros_like(labels, dtype=torch.float)
        mask[:, prompt_len-1:] = 1.0 
        
        # Advantages
        rewards_view = rewards_tensor.view(B, G)
        mean = rewards_view.mean(dim=1, keepdim=True)
        std = rewards_view.std(dim=1, keepdim=True) + 1e-8
        adv = (rewards_view - mean) / std
        adv = adv.view(-1).unsqueeze(1).repeat(1, labels.size(1))
        
        # KL Penalty
        with torch.no_grad():
            ref_logits = self.ref_model(generated_seqs)[:, :-1, :]
            ref_log_probs = torch.gather(ref_logits.log_softmax(dim=-1), -1, labels.unsqueeze(-1)).squeeze(-1)
            
        ratio_log = old_log_probs - ref_log_probs
        kl_penalty = CONFIG["beta"] * (torch.exp(ratio_log) - ratio_log - 1)
        
        # Loss
        loss_val = self.grpo.ppo_loss(old_log_probs, old_log_probs, adv, kl_penalty, mask)
        
        self.optimizer.zero_grad()
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Safety clip
        self.optimizer.step()
        
        return loss_val.item(), avg_reward

    def train(self):
        print("Starting GRPO Training (Audited Version)...")
        dataset = MusicRLVRDataset(["data/cantus_firmus_dataset.txt"], self.tokenizer)
        
        if len(dataset) == 0:
            print("Dataset empty! Run generate_data.py first.")
            return

        # Split Dataset (90/10)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        print(f"Data Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}")
        
        global_step = 0
        for epoch in range(CONFIG["total_epochs"]):
            # --- TRAINING LOOP ---
            self.model.train()
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            
            for batch in pbar:
                cf_ids = batch.to(self.device)
                loss, avg_r = self.train_step(cf_ids)
                
                # Logging
                self.writer.add_scalar("Train/Loss", loss, global_step)
                self.writer.add_scalar("Train/Reward", avg_r, global_step)
                
                pbar.set_postfix({"Loss": f"{loss:.2f}", "R": f"{avg_r:.2f}"})
                global_step += 1
            
            # --- VALIDATION LOOP ---
            if len(val_dataset) > 0:
                self.evaluate(val_loader, epoch)

            # Checkpoint
            if epoch % 10 == 0:
                 torch.save(self.model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], f"model_{epoch}.pt"))

    def evaluate(self, val_loader, epoch):
        """
        Runs a dry-run generation on validation set to check rewards.
        No gradients.
        """
        self.model.eval()
        total_reward = 0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                cf_ids = batch.to(self.device)
                G = CONFIG["group_size"]
                
                # Generate
                generated = self.model.generate_group(cf_ids, 16, group_size=G)
                
                # Score
                prompt_len = cf_ids.size(1)
                full_seqs = generated.tolist()
                
                for seq in full_seqs:
                    cf_part = seq[:prompt_len]
                    cp_part = seq[prompt_len:]
                    r = self.examiner.calculate_reward(cf_part, cp_part)
                    total_reward += r
                    count += 1
                    
        avg_val_reward = total_reward / count if count > 0 else 0
        print(f"Validation Reward: {avg_val_reward:.2f}")
        self.writer.add_scalar("Val/Reward", avg_val_reward, epoch)
            
            # Checkpoint
            if epoch % 10 == 0:
                 torch.save(self.model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], f"model_{epoch}.pt"))

if __name__ == "__main__":
    trainer = MusicGRPOTrainer()
    # trainer.train()
