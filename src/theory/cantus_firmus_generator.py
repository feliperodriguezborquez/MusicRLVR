import random
import os
from typing import List, Optional, Set, Tuple

class CantusFirmusGenerator:
    """
    Generates Cantus Firmus melodies based on strict Fuxian rules.
    Highly configurable via constructor arguments.
    """
    

    def __init__(self, config_path: str = None, **kwargs):
        """
        Initialize with a config file (JSON or YAML) or override with kwargs.
        """
        import json
        import yaml # Requires PyYAML
        
        # Default Config
        self.config = {
            "global_constraints": {
                "min_length": 8, "max_length": 16, "max_range": 16,
                "max_leaps": 3, "max_consecutive_dir_changes": 1,
                "min_dist_leap_from_ends": 2,
                "allow_repeated_notes": False, "allow_consecutive_leaps": False
            },
            "scale_rules": {
                "mode": "minor", "tonic": 62, "allow_plagal_range": False
            },
            "struct_rules": {
                "lowest_note": {
                    "limit": "tonic", "singularity_prob": 0.8, "approach_step_prob": 0.9
                },
                "climax": {
                    "avoid_start_measures": 3, "avoid_end_measures": 3,
                    "recovery_step_forced": True, "avoid_leading_tone_prob": 0.9,
                    "max_range_from_lowest": 16
                },
                "tritone_boundary_check": True
            },
            "probabilities": {
                "prefer_step_motion": 0.7, "allow_leap_prob": 0.3
            }
        }
        
        # Load File if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    loaded = yaml.safe_load(f)
                else:
                    loaded = json.load(f)
                    
                # Deep merge simplistically
                if loaded:
                    for k, v in loaded.items():
                        if k in self.config and isinstance(v, dict):
                            self.config[k].update(v)
                        else:
                            self.config[k] = v
        
        # Kwargs overrides (for backwards compatibility/scripting)
        # e.g. tonic, mode
        if 'tonic' in kwargs: self.config['scale_rules']['tonic'] = kwargs['tonic']
        if 'mode' in kwargs: self.config['scale_rules']['mode'] = kwargs['mode']
        if 'max_leaps' in kwargs: self.config['global_constraints']['max_leaps'] = kwargs['max_leaps']
        if 'max_consecutive_dir_changes' in kwargs: self.config['global_constraints']['max_consecutive_dir_changes'] = kwargs['max_consecutive_dir_changes']
        if 'min_dist_leap_from_ends' in kwargs: self.config['global_constraints']['min_dist_leap_from_ends'] = kwargs['min_dist_leap_from_ends']

        # Flatten strictly needed props for performance
        self.tonic = self.config['scale_rules']['tonic']
        self.mode = self.config['scale_rules']['mode']
        self.gc = self.config['global_constraints']
        
        # Determine range probabilities if present (User request: 0.8 octave, 0.2 tenth)
        # We look for 'max_range_prob' in probabilities section or strict struct_rules
        self.range_probs = self.config.get('probabilities', {}).get('max_range_dist', None)
        # Format assumed: {'octave': 0.8, 'tenth': 0.2}
        
        # Scale definitions
        if self.mode == 'major':
            self.scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        elif self.mode == 'minor':
            self.scale_intervals = [0, 2, 3, 5, 7, 8, 10]
        else:
            raise ValueError("Mode must be 'major' or 'minor'")
            
        # Allowed Melodic Intervals
        if 'melodic_rules' in self.config and 'allowed_intervals' in self.config['melodic_rules']:
             self.ALLOWED_INTERVALS = set(self.config['melodic_rules']['allowed_intervals'])
        else:
             self.ALLOWED_INTERVALS = {1, 2, 3, 4, 5, 7, 8, 12}

    def _get_scale_degree(self, note: int) -> int:
        norm = (note - self.tonic) % 12
        if norm in self.scale_intervals:
            return self.scale_intervals.index(norm)
        if self.mode == 'minor':
            if norm == 9: return 5 
            if norm == 11: return 6 
        return -1 

    def generate(self, num_samples: int = 1) -> List[List[int]]:
        results = []
        attempts = 0
        max_attempts = num_samples * 2000 
        
        while len(results) < num_samples and attempts < max_attempts:
            cf = self._generate_one()
            if cf:
                results.append(cf)
            attempts += 1
        return results

    def _generate_one(self) -> Optional[List[int]]:
        path = [self.tonic]
        
        # Dynamic Range Decision per Melody
        # Default global
        prob_range = self.gc['max_range']
        
        # Probabilistic Override
        if self.range_probs:
            r = random.random()
            # If 0.8 for octave (12), 0.2 for tenth (16)
            prob_octave = self.range_probs.get('octave', 0.8)
            if r < prob_octave:
                prob_range = 12
            else:
                prob_range = 16
        
        self.current_melody_range = prob_range # Store for use in recursion
        
        target_length = random.randint(self.gc['min_length'], self.gc['max_length'])
        return self._dfs(path, target_length)

    def _dfs(self, path: List[int], target_len: int) -> Optional[List[int]]:
        # Hard Constraints (Pruning)
        if self._count_leaps(path) > self.gc['max_leaps']: return None

        if len(path) == target_len:
            if self._validate_complete_melody(path):
                return path
            return None

        current_note = path[-1]
        candidates = self._get_candidates(current_note, path, target_len)
        
        # Probabilistic Filtering (Soft Rules)
        # Instead of just shuffle, we can sort/filter based on probabilities
        final_candidates = []
        for cand in candidates:
            if self._passes_soft_rules(path, cand, target_len):
                final_candidates.append(cand)
            else:
                # If failed soft rule, keep it with low probability?
                # User says "probabilidad asociada".
                # Let's say if it fails, we only keep it 10% of time.
                if random.random() < 0.1: 
                     final_candidates.append(cand)
        
        # Ensure we don't starve availability
        if not final_candidates and candidates:
             # Fallback: take some from original
             final_candidates = candidates[:1]
        
        random.shuffle(final_candidates)
        
        for next_note in final_candidates:
            res = self._dfs(path + [next_note], target_len)
            if res:
                return res
        return None

    def _count_leaps(self, path: List[int]) -> int:
        leaps = 0
        for i in range(len(path)-1):
            if abs(path[i+1] - path[i]) > 2:
                leaps += 1
        return leaps

    def _passes_soft_rules(self, path: List[int], candidate: int, target_len: int) -> bool:
        # Check against soft rules, return True/False based on probability configuration
        # If returns False, the candidate might still be picked with a very low fallback prob.
        
        sr = self.config['struct_rules']
        current_min = min(path)
        
        # 1. Lowest Note Rules (Singularity)
        if candidate == current_min:
            # EXCEPTION: If we are at the very last note (Cadence), we MUST allow Tonic (which is often min).
            is_final_note = len(path) == target_len - 1
            if is_final_note and candidate == self.tonic:
                 return True # Allow ending on tonic even if it repeats lowest note
            
            # Interior repetition of lowest note -> Check Probability
            # If random > prob, we FAIL (return False implies strict adherence if prob is high)
            # Confusing logic: simple check: if random < prob -> Fail?
            # Config says "singularity_prob: 0.8". Usually means "Prob of enforcing rule".
            # So if random < 0.8, we enforce (Fail).
            
            if random.random() < sr['lowest_note']['singularity_prob']: 
                return False
            
        # 2. Approach Lowest Note by Step
        # If candidate < current_min (new lowest) OR candidate == current_min (repeating it)
        # Actually rule says "Lowest note should be approached by step".
        # This applies when we CREATE the lowest note or RETURN to it.
        if candidate <= current_min:
            prev = path[-1]
            if abs(candidate - prev) > 2: # Leap
                 if random.random() < sr['lowest_note']['approach_step_prob']: 
                     return False

        # 3. Climax Rules
        # If this is a new potential climax
        current_max = max(path)
        if candidate > current_max:
             # Avoid Leading Tone Climax
             if self.mode == 'minor': # Check if Leading Tone
                  norm = (candidate - self.tonic) % 12
                  if norm == 11:
                       if random.random() < sr['climax']['avoid_leading_tone_prob']: 
                           return False
        
        return True

    def _get_candidates(self, current_note: int, path: List[int], target_len: int) -> List[int]:
        candidates = []
        
        # Range constraints
        # Authentic vs Plagal bounds
        min_p = self.tonic
        if self.config['scale_rules']['allow_plagal_range']:
             min_p = self.tonic - 5
        
        # Max bounds (global range)
        # Use simple relative to tonic + range check
        # But wait, true range is (Highest - Lowest).
        # We don't know lowest yet (could be plagal).
        # But we do know path so far. 
        # For simplicity, we limit relative to min_p and total allowed range.
        
        max_p = min_p + self.current_melody_range 
        
        # Optimization: Scan nearby valid notes
        search_range = range(min_p, self.tonic + 17) # approx 12th up
        
        for n in search_range:
            interval = abs(n - current_note)
            
            if interval == 0 and not self.gc['allow_repeated_notes']: continue
            if interval not in self.ALLOWED_INTERVALS: continue
            
            deg = self._get_scale_degree(n)
            if deg == -1: continue 
            
            candidates.append(n)

        # Filter candidates based on HARD local history
        valid = []
        for n in candidates:
            if self._passes_local_rules(path, n, target_len):
                valid.append(n)
        return valid

    def _passes_local_rules(self, path: List[int], next_note: int, target_len: int) -> bool:
        current_note = path[-1]
        interval = abs(next_note - current_note)
        direction = 1 if next_note > current_note else -1
        
        # 0. Min Dist Leap Ends
        if interval > 2:
            if len(path) <= self.gc['min_dist_leap_from_ends']: return False
            steps_remaining = target_len - len(path)
            if steps_remaining <= self.gc['min_dist_leap_from_ends']: return False

        # 1. Consecutive Leaps
        if len(path) > 1 and not self.gc['allow_consecutive_leaps']:
            prev_note = path[-2]
            prev_interval = abs(current_note - prev_note)
            if prev_interval > 2 and interval > 2: return False
        
        # 2. Max Consecutive Steps in Same Direction
        # Avoid scalar runs that are too long (e.g. C-D-E-F-G-A-B-C) if limit is low
        max_steps = self.config['global_constraints'].get('max_consecutive_steps_same_dir', 5)
        
        # Determine direction and type of current move
        diff = next_note - path[-1]
        is_step = abs(diff) <= 2
        direction_next = 1 if diff > 0 else -1 # Renamed to avoid conflict with 'direction'
        
        if is_step:
            consecutive_steps = 0
            # Look backwards
            for i in range(len(path) - 1, 0, -1):
                prev_diff = path[i] - path[i-1]
                prev_is_step = abs(prev_diff) <= 2
                
                if prev_diff == 0: break # Should be covered by repeated notes rule, but safety
                prev_dir = 1 if prev_diff > 0 else -1
                
                if prev_is_step and prev_dir == direction_next:
                    consecutive_steps += 1
                else:
                    break
            
            # If adding this one exceeds limit
            # We have 'consecutive_steps' existing. Adding 1 more.
            if consecutive_steps + 1 > max_steps:
                return False

        # 3. Large leap recovery (Strict)
        if len(path) > 1:
            prev_note = path[-2]
            prev_interval = abs(current_note - prev_note)
            prev_dir = 1 if current_note > prev_note else -1
            if prev_interval >= 5: 
                if interval > 2: return False 
                if direction == prev_dir: return False 

        # 3. Direction Changes
        dirs = []
        for i in range(len(path)-1):
            d = 1 if path[i+1] > path[i] else -1
            dirs.append(d)
        dirs.append(direction)
        
        cons_changes = 0
        if len(dirs) >= 2:
            for i in range(len(dirs)-1, 0, -1):
                if dirs[i] != dirs[i-1]: cons_changes += 1
                else: break
        
        if cons_changes > self.gc['max_consecutive_dir_changes']:
            # Exception: Leap Recovery
            prev_note_val = path[-2]
            current_note_val = path[-1]
            prev_int_size = abs(current_note_val - prev_note_val)
            if cons_changes == (self.gc['max_consecutive_dir_changes'] + 1):
                 if prev_int_size <= 2: return False
            else:
                 return False

        # 5. Minor specific
        if self.mode == 'minor':
            curr_norm = (current_note - self.tonic) % 12
            next_norm = (next_note - self.tonic) % 12
            
            # Ban Aug 2nd
            if curr_norm == 8 and next_norm == 11: return False
            if curr_norm == 11 and next_norm == 8: return False
            
            # Raised 6 -> Raised 7
            if curr_norm == 9 and next_norm != 11: return False
            
            # Climax Return Rule (Look-back) (Strict)
            # "Si llegas [al climax] por salto... debes bajar inmediatamente por grado"
            # We can't know if current_note IS the global climax yet.
            # But we can apply: "If I leaped UP to X, and X is a local peak, I must step down."
            # path[-2] (prev) --leap--> path[-1] (curr).
            # If interval > 2 AND direction == 1 (Up).
            # next_note MUST be step down.
            if len(path) > 1:
                 prev_note = path[-2]
                 prev_int = abs(current_note - prev_note)
                 prev_dir = 1 if current_note > prev_note else -1
                 
                 # If we leaped UP
                 if prev_int > 2 and prev_dir == 1:
                      # Check if next_note is step down
                      # Next MUST be down (recovery rule handles direction)
                      # Next MUST be step (<=2).
                      if interval > 2: return False

        return True

    def _validate_complete_melody(self, path: List[int]) -> bool:
        if path[0] != self.tonic or path[-1] != self.tonic: return False
        
        # Penultimate step
        if abs(path[-1] - path[-2]) > 2: return False
        
        high_point = max(path)
        low_point = min(path)
        
        # Singularity
        if path.count(high_point) > 1: return False
        
        # Climax Location
        max_idx = path.index(high_point)
        avoid_start = self.config['struct_rules']['climax']['avoid_start_measures']
        avoid_end = self.config['struct_rules']['climax']['avoid_end_measures']
        
        if max_idx < avoid_start: return False
        if max_idx >= len(path) - avoid_end: return False
        
        # Range check
        if (high_point - low_point) > self.current_melody_range: return False
        
        # Tritone Bound Check
        if self.config['struct_rules']['tritone_boundary_check']:
             # Check interval between low and high
             semitones = high_point - low_point
             if semitones == 6: return False # Exact tritone range
             # Or tritone relation modulo 12?
             if semitones % 12 == 6: return False
             
        return True

if __name__ == "__main__":
    print("Generating D Minor CFs with strict rules from JSON...")
    # Path assumes running from root
    import os
    cfg = os.path.join("src", "theory", "rules.json")
    gen = CantusFirmusGenerator(config_path=cfg, tonic=62, mode='minor')
    cfs = gen.generate(4)
    for cf in cfs:
        print(cf)

