import music21
import os
import tqdm

def download_and_process_bach():
    """
    Downloads JSB Chorales via music21, extracts outer voices (Soprano/Bass),
    and saves them in a simplified ABC-like token format for MusicRLVR.
    """
    print("Loading JSB Chorales from music21 corpus...")
    # music21's bach corpus contains 4-part chorales
    bach_corpus = music21.corpus.getComposer('bach')
    
    output_dir = os.path.join("data", "processed_bach")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    errors = 0
    
    # We will iterate through a subset to avoid overwhelming if corpus is huge,
    # or just use the standard 'chorals' if accessible.
    # listing pieces
    print("Finding chorales...")
    # search(query, field='composer') or similar. 
    # Simpler: Get paths directly
    chorales = music21.corpus.getComposer('bach')
    # Filter for xml/mxl, handling Path objects
    chorales = [str(c) for c in chorales if str(c).endswith('xml') or str(c).endswith('mxl')]
    
    output_file = os.path.join("data", "bach_soprano_bass.txt")
    
    with open(output_file, 'w') as f:
        print(f"Processing {len(chorales)} pieces...")
        
        for item_path in tqdm.tqdm(chorales):
            try:
                score = music21.converter.parse(item_path)
                
                # We need exactly 4 parts usually, but let's try to find Soprano and Bams
                # Usually Part 0 is Soprano, Part 3 (last) is Bass in standard SATB scores
                parts = score.parts
                if len(parts) < 2:
                    continue
                
                soprano = parts[0].flatten().notesAndRests
                bass = parts[-1].flatten().notesAndRests
                
                # Extract MIDI pitches (ignore rhythm for MVP Phase 1 simplicity, or keep if capable)
                # For MusicRLVR MVP we are focusing on PITCH sequences.
                
                s_pitches = []
                b_pitches = []
                
                # Simple extraction: just the sequence of pitches. 
                # Note: Dealing with rhythm synchronization is complex. 
                # For this MVP strict Counterpoint logic, we often assume 1:1 "First Species" or simple rhythm.
                # Bach chorales are homophonic but have passing notes.
                # We will just take the sequence of notes.
                
                for n in soprano.recurse().notes:
                    if n.isNote:
                        s_pitches.append(n.pitch.midi)
                    elif n.isChord:
                        s_pitches.append(n.root().midi) # Simplified
                        
                for n in bass.recurse().notes:
                    if n.isNote:
                        b_pitches.append(n.pitch.midi)
                    elif n.isChord:
                        b_pitches.append(n.root().midi)

                if len(s_pitches) < 8 or len(b_pitches) < 8:
                    continue

                # Format: <STYLE_REAL> <COMPOSER_BACH> [BASS_SEQ] [SOPRANO_SEQ]
                # In RLVR "Counterpoint", usually BASS is Cantus Firmus (Input) and Soprano is Counterpoint (Target), or vice versa.
                # Let's save both formats? Or just one. 
                # Let's write standard format: 
                # INPUT: <CF> note note ... TARGET: <CP> note note ...
                
                # Writing as abstract tokens
                s_str = " ".join(map(str, s_pitches))
                b_str = " ".join(map(str, b_pitches))
                
                # Line: <STYLE_REAL> <CF> bass_sequence <CP> soprano_sequence
                # We treat Bass as CF (fixed) and Soprano as CP (to be generated)
                f.write(f"<STYLE_REAL> <CF> {b_str} <CP> {s_str}\n")
                
                processed_count += 1
                
            except Exception as e:
                # print(f"Error processing {item}: {e}")
                errors += 1
                continue

    print(f"Finished. Processed {processed_count} chorales. Errors: {errors}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    download_and_process_bach()
