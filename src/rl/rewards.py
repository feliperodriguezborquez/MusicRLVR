from typing import List, Tuple, Set

class CounterpointRules:
    """
    The 'Examiner' for 1st Species Counterpoint.
    Evaluates a Counterpoint (CP) line against a Cantus Firmus (CF).
    """
    def __init__(self):
        # Perfect cons: 1(0), 5(7), 8(12)
        # Imperfect cons: 3(3,4), 6(8,9), 10(15,16) ...
        # Dissonances: 2, 4, 7...
        
        # Valid intervals (modulo 12)
        # Unison (0), Min3(3), Maj3(4), Perf5(7), Min6(8), Maj6(9)
        # Note: Perf4 (5) is dissonance in 2-part counterpoint!
        self.CONSONANCES = {0, 3, 4, 7, 8, 9} 
        self.PERFECT_CONS = {0, 7} # Modulo 12
        
        # We need absolute values for spacing checks too
        
    def calculate_reward(self, cf: List[int], cp: List[int]) -> float:
        """
        Calculates the total reward for a complete solution.
        Weighted sum of rule violations.
        """
        if len(cp) != len(cf):
            return -100.0 # Length mismatch is fatal
        
        score = 0.0
        
        # 1. Vertical Intervals (Note vs Note)
        for i in range(len(cp)):
            interval = abs(cp[i] - cf[i]) % 12
            if interval not in self.CONSONANCES:
                score -= 10.0 # Dissonance penalty
            
            # Distance constraint (max 10th or 12th usually)
            dist = abs(cp[i] - cf[i])
            if dist > 19: # > 12th + octave? Let's say max 2 octaves
                score -= 2.0
                
        # 2. Start/End Rules
        # Must start on perfect consonance (usually Octave or Unison or 5th)
        start_int = abs(cp[0] - cf[0]) % 12
        if start_int not in self.PERFECT_CONS:
             score -= 5.0
             
        # Must end on Perfect Consonance (usually Octave or Unison)
        end_int = abs(cp[-1] - cf[-1]) % 12
        if end_int != 0: # 8ve or Unison is 0 mod 12
            score -= 5.0

        # 3. Horizontal Motion (Voice Leading)
        for i in range(len(cp) - 1):
            u = cp[i]
            v = cp[i+1]
            cf_u = cf[i]
            cf_v = cf[i+1]
            
            # Melodic Interval
            mel_int = abs(v - u)
            if mel_int > 12: score -= 5.0 # Too big leap
            if mel_int in [6, 10, 11]: score -= 5.0 # Tritone, 7th leaps forbidden
            
            # Parallel Fifths/Octaves
            # If current vertical is 5/8 and next vertical is 5/8 
            # AND motion is parallel (not contrary)
            
            vert_1 = abs(u - cf_u)
            vert_2 = abs(v - cf_v)
            
            is_perfect_1 = (vert_1 % 12) in self.PERFECT_CONS
            is_perfect_2 = (vert_2 % 12) in self.PERFECT_CONS
            
            # Check parallel motion
            dir_cp = 1 if v > u else (-1 if v < u else 0)
            dir_cf = 1 if cf_v > cf_u else (-1 if cf_v < cf_u else 0)
            
            is_parallel = (dir_cp == dir_cf) and (dir_cp != 0)
            
            if is_parallel and is_perfect_1 and is_perfect_2:
                 # Check if it's the SAME interval class (Parallel) 
                 # e.g. 5th -> 5th or 8ve -> 8ve
                 if (vert_1 % 12) == (vert_2 % 12):
                     score -= 20.0 # Cardinal Sin!
                     
            # Direct Fifths/Octaves (Hidden)
            # Similar motion to a perfect consonance + Soprano leaps
            if is_parallel and is_perfect_2:
                 # If Soprano (CP) leaps
                 if mel_int > 2:
                      score -= 5.0
                      
        return score

    def calculate_step_reward(self, current_cf_note: int, current_cp_note: int, 
                              prev_cf_note: int = None, prev_cp_note: int = None) -> float:
        """
        Step-wise reward for RL environment (Immediate feedback).
        """
        r = 0.0
        
        # 1. Vertical
        interval = abs(current_cp_note - current_cf_note) % 12
        if interval in self.CONSONANCES:
            r += 1.0
        else:
            r -= 5.0 # Punishment for dissonance
            
        # 2. Voice Leading (requires previous state)
        if prev_cf_note is not None and prev_cp_note is not None:
            # Parallel 5/8 check
            vert_prev = abs(prev_cp_note - prev_cf_note)
            vert_curr = abs(current_cp_note - current_cf_note)
            
            if (vert_curr % 12) in self.PERFECT_CONS and (vert_prev % 12) == (vert_curr % 12):
                # Check direction
                dir_cp = current_cp_note - prev_cp_note
                dir_cf = current_cf_note - prev_cf_note
                if (dir_cp > 0 and dir_cf > 0) or (dir_cp < 0 and dir_cf < 0):
                    r -= 20.0 # Huge penalty
        
        return r
