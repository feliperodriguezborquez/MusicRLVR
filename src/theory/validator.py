from typing import List, Tuple, Optional

class CounterpointVerifier:
    """
    A class to verify First Species Counterpoint rules (Note against Note).
    Currently focuses on vertical intervals and parallel motion.
    
    Uses MIDI note numbers (integers) for robustness.
    Middle C = 60.
    """

    @staticmethod
    def get_interval(note1: int, note2: int) -> int:
        """Returns the absolute interval in semitones."""
        return abs(note1 - note2)

    @staticmethod
    def is_perfect_consonance(interval: int) -> bool:
        """
        Checks if interval is a perfect consonance (Unison, P5, or Octave equivalents).
        Unison: 0
        P5: 7
        P8: 12
        """
        interval = interval % 12
        return interval in [0, 7] 
        # Note: This is a simplified check. Pure P5 is 7. P4 is 5.
        # Strict counterpoint treats P4 as dissonant in 2-voice usually, 
        # but here we focus on detecting generic parallels first.
        # Actually, let's be more precise about parallels.

    @staticmethod
    def check_parallel_fifths(cantus_firmus: List[int], counterpoint: List[int]) -> List[Tuple[int, str]]:
        """
        Detects parallel fifths between two voices.
        Returns a list of errors: (index, description).
        
        Definition: Two voices move from a Perfect 5th to a Perfect 5th.
        Separated by step or leap, doesn't matter.
        """
        errors = []
        if len(cantus_firmus) != len(counterpoint):
            raise ValueError("Cantus Firmus and Counterpoint must have the same length for First Species.")

        # P5 is 7 semitones. (modulo 12)
        # We must check the actual interval size to avoid confusing P5 (7) with P4 (5).
        # We assume counterpoint is usually above CF, but we should handle both.
        
        for i in range(len(cantus_firmus) - 1):
            u1, l1 = counterpoint[i], cantus_firmus[i]
            u2, l2 = counterpoint[i+1], cantus_firmus[i+1]
            
            interval_1 = abs(u1 - l1)
            interval_2 = abs(u2 - l2)
            
            # Check if both start and end intervals are effectively Perfect 5ths (7, 19, 31...)
            # We assume simple integer checking for now.
            is_p5_1 = (interval_1 % 12 == 7)
            is_p5_2 = (interval_2 % 12 == 7)
            
            if is_p5_1 and is_p5_2:
                # Check if there is motion (if notes stay same, it's repeated notes, not strict parallel motion error usually, but strict CP forbids static oblique if not tied?)
                # Actually strict CP parallel 5ths applies even if repeated? No, repetition is allowed but weak.
                # Usually "Parallel" implies motion.
                if u1 != u2 or l1 != l2:
                    errors.append((i, f"Parallel Fifth detected at index {i}->{i+1}: ({l1},{u1}) -> ({l2},{u2})"))

        return errors

    @staticmethod
    def check_parallel_octaves(cantus_firmus: List[int], counterpoint: List[int]) -> List[Tuple[int, str]]:
        """
        Detects parallel octaves (or unisons) between two voices.
        """
        errors = []
        if len(cantus_firmus) != len(counterpoint):
            raise ValueError("Cantus Firmus and Counterpoint must have the same length.")

        for i in range(len(cantus_firmus) - 1):
            u1, l1 = counterpoint[i], cantus_firmus[i]
            u2, l2 = counterpoint[i+1], cantus_firmus[i+1]
            
            interval_1 = abs(u1 - l1)
            interval_2 = abs(u2 - l2)
            
            # Check for Octaves (0, 12, 24...)
            is_p8_1 = (interval_1 % 12 == 0)
            is_p8_2 = (interval_2 % 12 == 0)
            
            if is_p8_1 and is_p8_2:
                if u1 != u2 or l1 != l2:
                   errors.append((i, f"Parallel Octave/Unison detected at index {i}->{i+1}: ({l1},{u1}) -> ({l2},{u2})"))
                   
        return errors

    ###########################################################################
    # HELPER TO RUN CHECKS
    ###########################################################################
    def validate(self, cantus_firmus: List[int], counterpoint: List[int]) -> bool:
        """
        Runs all checks and prints errors. Returns True if valid.
        """
        fifths = self.check_parallel_fifths(cantus_firmus, counterpoint)
        octaves = self.check_parallel_octaves(cantus_firmus, counterpoint)
        
        all_errors = fifths + octaves
        
        if not all_errors:
            print("[OK] No parallel fifths or octaves detected.")
            return True
        else:
            print(f"[FAIL] Found {len(all_errors)} errors:")
            for index, msg in sorted(all_errors):
                print(f"  - Step {index}: {msg}")
            return False

# Example usage/Test
if __name__ == "__main__":
    verifier = CounterpointVerifier()
    
    # Example 1: D Minor (Dorianish)
    # CF: D F E D
    cf = [62, 65, 64, 62] 
    
    # User Attempt (Parallel 5ths)
    # CP: A C B A (perfect 5ths above)
    cp_bad = [69, 72, 71, 69]
    
    print("Testing Bad Counterpoint (Parallel 5ths):")
    verifier.validate(cf, cp_bad)
    
    # Valid attempt
    # CP: F A G F (3rds/6ths ?) 
    # D-F (min3), F-A (maj3), E-G (min3), D-F (min3) - Parallel 3rds are generally OK in moderation
    cp_ok = [65, 69, 67, 65]
    print("\nTesting Good Counterpoint:")
    verifier.validate(cf, cp_ok)
