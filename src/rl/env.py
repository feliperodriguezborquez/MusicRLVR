from src.rl.rewards import CounterpointRules
import torch

class CounterpointEnv:
    """
    Simulates the environment.
    State: (CF_Note_Index, [History of CP notes], Current_CF_Melody)
    Action: Next Note (Midi int or Token ID)
    """
    def __init__(self, cf_line: list):
        self.cf_line = cf_line # List of MIDI input
        self.cp_line = []      # List of MIDI output
        self.idx = 0
        self.rules = CounterpointRules()
        self.done = False
        
    def reset(self):
        self.cp_line = []
        self.idx = 0
        self.done = False
        return self._get_state()
        
    def _get_state(self):
        # MVP State: Just the input CF + current generated CP so far
        # For a Transformer, the state is the entire sequence tokens.
        return (self.cf_line, self.cp_line)
        
    def step(self, action_note: int):
        if self.done:
            return self._get_state(), 0, True, {}
            
        self.cp_line.append(action_note)
        
        # Calculate Reward
        # We need context.
        current_cf = self.cf_line[self.idx]
        prev_cf = self.cf_line[self.idx-1] if self.idx > 0 else None
        prev_cp = self.cp_line[-2] if len(self.cp_line) > 1 else None
        
        reward = self.rules.calculate_step_reward(
            current_cf, action_note, prev_cf, prev_cp
        )
        
        self.idx += 1
        if self.idx >= len(self.cf_line):
            self.done = True
            # Optional: Add final full-sequence reward check
            full_reward = self.rules.calculate_reward(self.cf_line, self.cp_line)
            # Add difference to avoid double counting? Or just use terminal reward.
            # Usually strict RL uses dense rewards (step) + sparse terminal.
            # Let's add the global consistency check penalty/bonus at the end.
            reward += full_reward
            
        return self._get_state(), reward, self.done, {}
