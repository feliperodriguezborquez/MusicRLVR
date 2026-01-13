
import os
import random
from src.theory.cantus_firmus_generator import CantusFirmusGenerator

def generate_dataset():

    # Path to rules.yaml
    rules_path = os.path.join("src", "theory", "rules.yaml")
    
    # Strict Configuration for High Quality Data
    # We use the YAML file for detailed rules.
    # We want a diverse dataset, so we will generate batches in different keys.
    
    dataset_size = 5000 
    # Create lists of interesting tonics (middle range)
    # Minor Keys: A, B, C, C#, D, E, F, G (MIDI 57-67)
    # Major Keys: C, D, E, F, G, A, Bb (MIDI 60-70)
    
    minor_tonics = [57, 59, 60, 61, 62, 64, 65, 67]
    major_tonics = [60, 62, 64, 65, 67, 69, 70]
    
    all_cfs = []
    
    batch_size = dataset_size // (len(minor_tonics) + len(major_tonics)) + 1
    
    print(f"Generating diverse dataset (Goal: {dataset_size})...")
    
    # 1. Generate Minors
    for t in minor_tonics:
        gen = CantusFirmusGenerator(config_path=rules_path, tonic=t, mode='minor')
        batch = gen.generate(batch_size)
        all_cfs.extend(batch)
        print(f"Generated {len(batch)} samples for Minor tonic {t}")

    # 2. Generate Majors
    for t in major_tonics:
        gen = CantusFirmusGenerator(config_path=rules_path, tonic=t, mode='major')
        batch = gen.generate(batch_size)
        all_cfs.extend(batch)
        print(f"Generated {len(batch)} samples for Major tonic {t}")
        
    random.shuffle(all_cfs)
    cfs = all_cfs[:dataset_size] # Trim to exact size
    
    output_path = os.path.join("data", "cantus_firmus_dataset.txt")
    
    with open(output_path, "w") as f:
        for cf in cfs:
            # Format: <STYLE_EXERCISE> <CF> note note ...
            # We convert list of ints to space-separated string
            cf_str = " ".join(map(str, cf))
            f.write(f"<STYLE_EXERCISE> <CF> {cf_str}\n")
            
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_dataset()
