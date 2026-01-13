
import music21
import verovio
import sys
import ast
import os
import random
import re

def visualize_melodies(melodies_data, titles=None):
    """
    Render melodies to separate SVG files using Verovio.
    """
    
    normalized_data = []
    if isinstance(melodies_data[0], list) and isinstance(melodies_data[0][0], int):
        for m in melodies_data:
            normalized_data.append({'melody': m, 'tonic': 62, 'mode': 'minor'})
    else:
        normalized_data = melodies_data

    output_paths = []

    for i, data in enumerate(normalized_data):
        # Create a FRESH Score for each melody
        score = music21.stream.Score()
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = "" 
        score.metadata.composer = ""
        
        main_part = music21.stream.Part()
        main_part.id = "MainPart"
        
        melody_ints = data['melody']
        tonic = data.get('tonic', 62)
        mode = data.get('mode', 'minor')
        
        # Title logic
        pitch_name = music21.pitch.Pitch(tonic).name.replace('-', 'b')
        mode_suffix = "m" if mode == 'minor' else ""
        short_title = f"{pitch_name}{mode_suffix}"
            
        # Key
        tonic_pitch = music21.pitch.Pitch(tonic)
        key = music21.key.Key(tonic_pitch.name, mode)
        
        # Determine number of measures to set width dynamically if needed
        # But we'll stick to fixed page for consistency
        
        for j, pitch_int in enumerate(melody_ints):
            m = music21.stream.Measure()
            m.number = j + 1
            
            # Start of Melody Setup
            if j == 0:
                m.append(music21.clef.TrebleClef())
                m.append(key)
                # Add Label (Title)
                te = music21.expressions.TextExpression(short_title)
                te.placement = 'above'
                te.style.fontWeight = 'bold'
                te.style.fontSize = 14
                m.append(te)

            n = music21.note.Note(pitch_int)
            
            # FIX ENHARMONICS
            if key.sharps > 0:
                if n.pitch.accidental and n.pitch.accidental.name == 'flat':
                    enh = n.pitch.getEnharmonic()
                    if enh.accidental.name != 'flat': n.pitch = enh
            elif key.sharps < 0:
                 if n.pitch.accidental and n.pitch.accidental.name == 'sharp':
                     enh = n.pitch.getEnharmonic()
                     if enh.accidental.name != 'sharp': n.pitch = enh
            
            n.duration.type = 'whole'
            
            # Hide redundant accidentals
            step = n.pitch.step
            expected_acc = key.accidentalByStep(step)
            exp_str = expected_acc.name if expected_acc else 'natural'
            curr_str = n.pitch.accidental.name if n.pitch.accidental else 'natural'
            
            if exp_str == curr_str:
                if n.pitch.accidental: n.pitch.accidental.displayStatus = False
            else:
                 if n.pitch.accidental: n.pitch.accidental.displayStatus = True
            
            m.append(n)
            
            if j == len(melody_ints) - 1:
                m.rightBarline = music21.bar.Barline('light-heavy')
                
            main_part.append(m)
        
        score.append(main_part)
        
        # Export XML
        out_xml = music21.musicxml.m21ToXml.GeneralObjectExporter(score).parse()
        xml_str = out_xml.decode('utf-8')
        
        # Cleanup XML
        xml_str = re.sub(r'<time>.*?</time>', '', xml_str, flags=re.DOTALL)
        xml_str = re.sub(r'<part-group.*?>.*?</part-group>', '', xml_str, flags=re.DOTALL)
        
        # Render
        tk = verovio.toolkit()
        tk.setOptions({
            "pageWidth": 2500, # Wide landscape
            "pageHeight": 500, # Short height for single strip
            "scale": 35,
            "adjustPageHeight": True,
            "font": "Bravura",
            "header": "none",
            "footer": "none"
        })
        
        tk.loadData(xml_str)
        svg_data = tk.renderToSVG(1)
        
        if "<svg" in svg_data:
            svg_data = svg_data.replace("<svg", '<svg style="background-color:white"')
        
        filename = f"score_example_{i+1}.svg"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(svg_data)
        
        output_paths.append(filename)
        print(f"Saved {filename}")

    return output_paths

if __name__ == "__main__":
    from src.theory.cantus_firmus_generator import CantusFirmusGenerator
    
    rules_path = os.path.join("src", "theory", "rules.yaml")

    # We want varied keys for demo
    demos = []
    
    # 1. C# Minor
    print("Generating Example 1 (C# Minor)...")
    gen_csh = CantusFirmusGenerator(config_path=rules_path, tonic=61, mode='minor')
    cfs = gen_csh.generate(1)
    if cfs: demos.append({'melody': cfs[0], 'tonic': 61, 'mode': 'minor'})
    
    # 2. F Major
    print("Generating Example 2 (F Major)...")
    gen_f = CantusFirmusGenerator(config_path=rules_path, tonic=65, mode='major')
    cfs2 = gen_f.generate(1)
    if cfs2: demos.append({'melody': cfs2[0], 'tonic': 65, 'mode': 'major'})

    # 3. Random Minor (A Minor)
    print("Generating Example 3 (A Minor)...")
    gen_a = CantusFirmusGenerator(config_path=rules_path, tonic=57, mode='minor')
    cfs3 = gen_a.generate(1)
    if cfs3: demos.append({'melody': cfs3[0], 'tonic': 57, 'mode': 'minor'})

    visualize_melodies(demos)
