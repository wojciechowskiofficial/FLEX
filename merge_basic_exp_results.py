import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import os
from shutil import copy2


def Main():
    df = pd.read_csv("./metadata/val_paths_experiments_0.csv", header=None)
    image_names = df.iloc[:,0].tolist()

    # Explain
    for image_name in tqdm(image_names):
        full_image_path = os.path.join(args.val_set_parent_path, image_name)
        
        # Create "image folder"
        if not os.path.exists(os.path.join(args.output_parent_path, image_name[:-5])): # [:-5] is for getting rid of ".JPEG"
                os.makedirs(os.path.join(args.output_parent_path, image_name[:-5]))
                
        # Copy image to the "image folder"
        if not args.do_not_copy_image and not os.path.isfile(os.path.join(args.output_parent_path, image_name[:-5], image_name)):
            copy2(full_image_path, os.path.join(args.output_parent_path, image_name[:-5], image_name))
            
        # Copy MALE to the "image folder"
        copy2(os.path.join("merge_explanations/male/imagenet/resnet18", image_name[:-5], "soft_explanation.txt"), os.path.join(args.output_parent_path, image_name[:-5], image_name + "_male_soft_gpt4_explanation.txt"))
        
        # Copy show, tell & attend
        copy2(os.path.join("merge_explanations/showattendtell", image_name[:-5], image_name[:-5] + "_explanation.txt"), os.path.join(args.output_parent_path, image_name[:-5], image_name + "_showattendtell_explanation.txt"))
        
        # Copy nlx-gpt
        copy2(os.path.join("merge_explanations/nlx_gpt_explanations/imagenet", image_name[:-5], image_name[:-5] + "_nlx_gpt_exp.txt"), os.path.join(args.output_parent_path, image_name[:-5], image_name + "_nlx-gpt_explanation.txt"))
        
                
                
if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./various_method_explanations")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main()