import torch
import os
import random
import music21
import argparse
from src.models.policy import PolicyTransformer
from src.data.dataset import MusicTokenizer

def load_model(checkpoint_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MusicTokenizer()
    model = PolicyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128, nhead=4, num_layers=2
    ).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Using random weights.")
    model.eval()
    return model, tokenizer, device

def generate_counterpoint(model, tokenizer, device, cf_notes, species=1):
    # Construct Prompt: <SPECIES_X> <STYLE_EXERCISE> <CF> ... <CP>
    # Note: Training logic prepended Species token.
    cf_str = " ".join([str(n) for n in cf_notes])
    prompt_str = f"<SPECIES_{species}> <STYLE_EXERCISE> <CF> {cf_str} <CP>"
    
    print(f"Prompt: {prompt_str}")
    
    input_ids = tokenizer.encode(prompt_str)
    if input_ids[-1] == tokenizer.eos_id: input_ids = input_ids[:-1]
        
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Max length depends on species (S2 = 2x, S5 = 4x approx)
    ratio = 4 if species >= 3 else (2 if species == 2 else 1)
    max_new = (len(cf_notes) * ratio) + 16
    
    generated = model.generate_group(input_tensor, max_new_tokens=max_new, group_size=1, temperature=0.8)
    
    # Decode
    full_text = tokenizer.decode(generated[0].tolist())
    
    # Parse CP portion
    if "<CP>" in full_text:
        cp_section = full_text.split("<CP>")[1]
        cp_notes = [int(x) for x in cp_section.split() if x.isdigit()]
    else:
        cp_notes = []
            
    return cp_notes

def create_score(cf_notes, cp_notes, species, filename="solution"):
    score = music21.stream.Score()
    score.metadata = music21.metadata.Metadata(title=f"Species {species} Solution", composer="AI Agent")
    
    # CP (Soprano)
    cp_part = music21.stream.Part(id="Counterpoint")
    cp_part.append(music21.clef.TrebleClef())
    
    for i, n_int in enumerate(cp_notes):
        n = music21.note.Note(n_int)
        
        # Rhythm Logic
        if species == 1:
            n.duration.type = "whole"
        elif species == 2:
            # S2: Half notes, last is whole
            if i == len(cp_notes) - 1:
                n.duration.type = "whole"
            else:
                n.duration.type = "half"
        else:
            # Placeholder for S3/4/5
            n.duration.type = "quarter"
            
        cp_part.append(n)
        
    # CF (Bass)
    cf_part = music21.stream.Part(id="CantusFirmus")
    cf_part.append(music21.clef.BassClef())
    for n_int in cf_notes:
        n = music21.note.Note(n_int)
        n.duration.type = "whole" 
        cf_part.append(n)
        
    score.append(cp_part)
    score.append(cf_part)
    
    xml_path = f"{filename}.musicxml"
    score.write('musicxml', fp=xml_path)
    print(f"Saved Score: {xml_path}")
    score.write('midi', fp=f"{filename}.mid")

def run_demo(species):
    data_path = "data/cantus_firmus_dataset.txt"
    if not os.path.exists(data_path): return

    with open(data_path, "r") as f:
        line = random.choice(f.readlines()).strip()
    
    # Parse just notes from raw line
    # Raw: <STYLE> <CF> ...
    tokens = line.split()
    cf_notes = [int(t) for t in tokens if t.isdigit()]
    
    print(f"CF: {cf_notes} (Target Species {species})")
    
    ckpt_dir = "checkpoints"
    ckpt_path = None
    if os.path.exists(ckpt_dir):
        files = glob = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if files: ckpt_path = os.path.join(ckpt_dir, sorted(files)[-1])
            
    model, tokenizer, device = load_model(ckpt_path)
    cp_notes = generate_counterpoint(model, tokenizer, device, cf_notes, species)
    print(f"CP: {cp_notes}")
    
    create_score(cf_notes, cp_notes, species, filename=f"solution_s{species}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=int, default=1, help="Counterpoint Species (1-5)")
    args = parser.parse_args()
    
    run_demo(args.species)
