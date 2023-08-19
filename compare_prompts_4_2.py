import os
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import re
import ast
from collections import defaultdict


def Main(args):
    df = pd.read_csv(args.val_images_names_full_path, header=None)
    image_names = df.iloc[:,0].tolist()

    for image_name in tqdm(image_names):
        with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], "is_flip_True", "prompt.txt"), "r") as f:
             prompt = f.read()

        match = re.search(r'\[(\{.*?\})\]', prompt)

        if match:
            # Extract the dictionary string
            dict_string = match.group(1)
            
            # Convert the string representation to a Python dictionary
            dictionary_flipped_in_tuple = ast.literal_eval(dict_string)
            dictionary_flipped = defaultdict(lambda :[])
            for el in dictionary_flipped_in_tuple:
                dictionary_flipped[el["description"]] += el["positions"]
            dictionary_flipped = dict(dictionary_flipped)
        else:
            print("No dictionary found")

        
        # Og prompt read
        with open(os.path.join("basic_explanations", args.dataset, args.cnn, image_name[:-5], "prompt.txt"), "r") as f:
             prompt = f.read()

        # Match the pattern
        match = re.search(r'\[(\{.*?\})\]', prompt)

        if match:
            # Extract the dictionary string
            dict_string = match.group(1)
            
            # Convert the string representation to a Python dictionary
            dictionary_og_in_tuple = ast.literal_eval(dict_string)
            dictionary_og = defaultdict(lambda :[])
            for el in dictionary_og_in_tuple:
                dictionary_og[el["description"]] += el["positions"]
            dictionary_og = dict(dictionary_og)
        else:
            print("No flipped dictionary found")
        
        # Intersect keys
        intersected_set = set(dictionary_flipped.keys()).intersection(set(dictionary_og.keys()))

        left_og_count, right_og_count, left_flipped_count, right_flipped_count = 0, 0, 0, 0
        for k in intersected_set:
            left_og_count += dictionary_og[k].count("left")
            right_og_count += dictionary_og[k].count("right")
            
            left_flipped_count += dictionary_flipped[k].count("left")
            right_flipped_count += dictionary_flipped[k].count("right")
        print(left_og_count, right_og_count, left_flipped_count, right_flipped_count)
        print(dictionary_flipped)
        print(dictionary_og)
        print("###")
        


if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./experiment_4_2_results")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main(args)