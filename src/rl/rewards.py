import math

class CounterpointRules:
    """
    The Examiner.
    Evaluates a counterpoint solution against Fuxian rules.
    Supports Species 1-5.
    """
    def __init__(self):
        # Perfect Consonances: 1 (Unison), 5, 8, 12, ...
        self.perfect_consonances = {0, 7, 12, 19, 24}
        # Imperfect Consonances: 3, 4 (Maj3), 8, 9 (Maj6), ...
        # Simplified Interval set modulo 12:
        # 0: P1 (C), 3: m3 (C), 4: M3 (C), 7: P5 (C), 8: m6 (C), 9: M6 (C)
        self.consonances_mod12 = {0, 3, 4, 7, 8, 9}
        
    def calculate_reward(self, cf_notes, cp_notes, species=1):
        """
        Calculates the reward for a given solution.
        species: int (1-5) or 0 (Free/Bach)
        """
        if not cp_notes:
            return -10.0
            
        reward = 0.0
        
        # 1. Length/Structure Check
        if species == 1:
            reward += self._check_species_1_structure(cf_notes, cp_notes)
            reward += self._check_species_1_rules(cf_notes, cp_notes)
        elif species == 2:
            reward += self._check_species_2_structure(cf_notes, cp_notes)
            reward += self._check_species_2_rules(cf_notes, cp_notes)
        else:
            # Placeholder for S3, S4, S5
            # Just reward length matching implied ratio
            ratio = 4 if species == 3 else 1
            if len(cp_notes) >= len(cf_notes) * ratio:
                reward += 1.0
            reward += self._check_generic_harmony(cf_notes, cp_notes, species)

        return reward

    # --- SPECIES 1 (1:1) ---
    def _check_species_1_structure(self, cf, cp):
        r = 0.0
        if len(cp) != len(cf):
            r -= 5.0 # Wrong length
        
        # Start/End rules
        if cp and len(cp) > 0:
            # First vertical interval should be Perfect (1, 5, 8)
            iv = abs(cp[0] - cf[0]) % 12
            if iv in {0, 7}: r += 1.0
            else: r -= 1.0
        return r

    def _check_species_1_rules(self, cf, cp):
        r = 0.0
        length = min(len(cf), len(cp))
        
        for i in range(length):
            # Vertical Interval
            iv = abs(cp[i] - cf[i])
            iv_mod = iv % 12
            
            # Consonance
            if iv_mod in self.consonances_mod12:
                r += 1.0
            else:
                r -= 2.0 # Penalize dissonance in S1
                
            # Parallel 5ths/8ves
            if i > 0:
                prev_iv = abs(cp[i-1] - cf[i-1]) % 12
                curr_iv = iv_mod
                
                # If moving in parallel (simplified checks)
                if prev_iv == 7 and curr_iv == 7 and cp[i] != cp[i-1]:
                    r -= 20.0 # Huge Penalty
                if prev_iv == 0 and curr_iv == 0 and cp[i] != cp[i-1]:
                    r -= 20.0
                    
        return r

    # --- SPECIES 2 (2:1) ---
    def _check_species_2_structure(self, cf, cp):
        r = 0.0
        # Expected length: 2 * (len(cf) - 1) + 1 (last note is whole)
        # Usually S2 ends with a whole note.
        target_len_min = (len(cf) - 1) * 2
        
        if len(cp) < target_len_min:
            r -= 5.0
        elif len(cp) > target_len_min + 2:
            r -= 2.0 
        else:
            r += 2.0
            
        return r

    def _check_species_2_rules(self, cf, cp):
        r = 0.0
        # Map CP notes to CF notes
        # Ratio 2:1 means CP[0], CP[1] go with CF[0]
        # Last note cf[-1] goes with last note cp[-1]
        
        cf_idx = 0
        cp_idx = 0
        
        while cf_idx < len(cf) - 1 and cp_idx < len(cp) - 1:
            # Downbeat (Strong) -> Must be Consonant
            note_strong = cp[cp_idx]
            cf_note = cf[cf_idx]
            
            iv_strong = abs(note_strong - cf_note) % 12
            if iv_strong in self.consonances_mod12:
                r += 1.0
            else:
                r -= 5.0 # Dissonant downbeat illegal
                
            # Upbeat (Weak) -> Can be Dissonant if Passing Tone
            if cp_idx + 1 < len(cp):
                note_weak = cp[cp_idx + 1]
                iv_weak = abs(note_weak - cf_note) % 12
                
                if iv_weak in self.consonances_mod12:
                    r += 0.5 # Consonant weak beat is fine
                else:
                    # Check passing tone logic: Stepwise motion
                    # Previous note (strong), Current (weak), Next (strong of next bar)
                    prev_n = note_strong
                    curr_n = note_weak
                    # Next note is cp[cp_idx+2]
                    # This check is complex for MVP, simplified:
                    # If step motion (interval <= 2 semitones or so), minimal penalty
                    melodic_iv = abs(curr_n - prev_n)
                    if melodic_iv <= 2: 
                        r += 0.5 # Passing tone likely
                    else:
                        r -= 2.0 # Leaping into dissonance is bad
                        
            cf_idx += 1
            cp_idx += 2
            
        return r
        
    def _check_generic_harmony(self, cf, cp, species):
        # Basic consonance check for other species
        r = 0.0
        # Sample points to estimate quality
        return r
