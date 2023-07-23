import subprocess
from caption import caption_image_beam_search, visualize_att
import torch
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORD_MAP_PATH = "/home/adamwsl/MALE/captioning/a-PyTorch-Tutorial-to-Image-Captioning/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
MODEL_PATH = "/home/adamwsl/MALE/captioning/a-PyTorch-Tutorial-to-Image-Captioning/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
VAL_SET_PATH = "./val"
OUTPUT_DIR = './experiment_1_results'

def process_single_image(img_path: str) -> Tuple[str, plt.figure]:
     # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(WORD_MAP_PATH, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, img_path, word_map, beam_size=5)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    exp, fig = visualize_att(img_path, seq, alphas, rev_word_map)
    return exp, fig


def Main():
    # Load CSV file
    df = pd.read_csv("val_paths_experiments_0.csv", header=None)
    partial_image_paths = df[0].str.split(",", n=1, expand=True)[0].tolist()
    
    # Start 
    for partial_image_path in partial_image_paths:
        # Produce exp for one image
        full_img_path = os.path.join(VAL_SET_PATH, partial_image_path)
        exp, fig = process_single_image(full_img_path)
        
        # Save the explanations
        output_subdir = os.path.join(OUTPUT_DIR, partial_image_path.split('.')[0])  # Strip extension
        os.makedirs(output_subdir, exist_ok=True)
        exp_dir = os.path.join(output_subdir, partial_image_path.split('.')[0] + "_explanation.txt")
        with open(exp_dir, "w") as f:
            f.write(exp)
        viz_dir = os.path.join(output_subdir, partial_image_path.split('.')[0] + "_viz.jpg")
        fig.savefig(viz_dir)
        

if __name__ == "__main__":
    Main()