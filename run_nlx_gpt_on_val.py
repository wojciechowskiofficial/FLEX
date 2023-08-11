import pandas as pd
import os
from tqdm import tqdm
import requests
import gradio as gr
from argparse import ArgumentParser
from shutil import copy2


def Main():
    
    df = pd.read_csv(args.val_images_names_full_path, header=None)
    image_names = df.iloc[:,0].tolist()
    
    # Explain
    for image_name in tqdm(image_names):
        full_image_path = os.path.join(args.val_set_parent_path, image_name)

        # To convert your image file into the base64 format required by the API
        encoded_image = gr.processing_utils.encode_url_or_file_to_base64(full_image_path)
            
        r = requests.post(url='https://fawaz-nlx-gpt.hf.space/api/predict/',json={"data":[encoded_image,"What is portrayed on this image?"]})
        what, why, _ = r.json()["data"]
        explanation = f"There is {what} in the image {why}."
        if not os.path.exists(os.path.join(args.output_parent_path, args.dataset,  image_name[:-5])): # [:-5] is for getting rid of ".JPEG"
            os.makedirs(os.path.join(args.output_parent_path, args.dataset, image_name[:-5]))
        with open(os.path.join(args.output_parent_path, args.dataset, image_name[:-5], "nlx_gpt_exp.txt"), "w") as f:
            f.write(explanation)
        if not args.do_not_copy_image and not os.path.isfile(os.path.join(args.output_parent_path, args.dataset, image_name[:-5], image_name)):
            copy2(full_image_path, os.path.join(args.output_parent_path, args.dataset, image_name[:-5], image_name))


if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./nlx_gpt_explanations")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main()