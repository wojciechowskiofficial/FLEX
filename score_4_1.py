from src.male import run_pipeline_single_decision
import os
import pandas as pd
from tqdm import tqdm
from numpy import save
from shutil import copy2
from argparse import ArgumentParser
from sacrebleu import sentence_bleu
from evaluate import load


def Main(args):    
    a = "Your string a herefsdaaaaaaaaaaaaaaaeeiajoiejfoaijeofijasiofjeioafjoeisahfiuejashhifgueahioufgheioashfgiouehjsaoifhjeoiashfoeishaho"
    b = 'The model identified this image as a "mosquito net" because it detected items with straight features at the top-right corner and right of the image. Additionally, it found a gusher of water and a ship, reflective surfaces, and red and blue objects throughout the entire image. The model also detected animals and screens at the top, top-right corner, center, and right of the image. Furthermore, it identified beds, items with circular features, and rounded edges in pictures throughout the entire image. Although the model did not find any human hands, it did detect animals and clothes in the upper half of the image.'

    # BLEU Score
    bleu = sentence_bleu(b, [a]).score
    print(f"BLEU Score: {bleu}")

    # BertSCORE
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=[a], references=[b], lang="en")
    print(results)

    # METEOR Score
    #meteor = load('meteor')
    #print(results)


    df = pd.read_csv(args.val_images_names_full_path, header=None)
    image_names = df.iloc[:,0].tolist()
    
    for noise_intensity in [.05]:
        variation = f"noise_int_{noise_intensity}"
        # Explain
        for image_name in tqdm(image_names):
            full_image_path = os.path.join(args.val_set_parent_path, image_name)
            
            with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation, "soft_explanation.txt"), "r") as f:
                noise_exp = f.read().strip()
            
            with open(os.path.join("basic_explanations", args.dataset, args.cnn, image_name[:-5], "soft_explanation.txt"), "r") as f:
                og_exp = f.read().strip()
            
            print(noise_exp, og_exp)
            break



if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./experiment_4_1_results")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main(args)