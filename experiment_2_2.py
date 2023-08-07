from src.male import run_pipeline_single_decision
import os
import pandas as pd
from tqdm import tqdm
from numpy import save, linspace
from shutil import copy2
from argparse import ArgumentParser


def Main(args):
    # Prepare for processing 
    if args.cnn == "resnet18":
        from torchvision.models import resnet18
        last_layer_name = "layer4"
        model = resnet18(pretrained=True)
        _ = model.eval()
        
        layer_map = {'conv1' : model.bn1, 
                    'layer1' : model.layer1[1].bn2, 
                    'layer2' : model.layer2[1].bn2, 
                    'layer3' : model.layer3[1].bn2, 
                    'layer4' : model.layer4[1].bn2}
    elif args.cnn == "alexnet":
        from torchvision.models import alexnet
        last_layer_name = "conv5"
        model = alexnet(pretrained=True)
        _ = model.eval()
        
        layer_map = {'conv1' : model.features[0], 
                     'conv2' : model.features[3], 
                     'conv3' : model.features[6], 
                     'conv4' : model.features[8], 
                     'conv5' : model.features[10]}
    
    df = pd.read_csv(args.val_images_names_full_path, header=None)
    image_names = df.iloc[:,0].tolist()
    
    for temperature in linspace(.1, 2., 5):
        variation = f"temp_{temperature}"
        for image_name in tqdm(image_names):
            full_image_path = os.path.join(args.val_set_parent_path, image_name)
            
            probabilities, prompt, explanation = run_pipeline_single_decision(model=model, 
                                                                            full_image_path=full_image_path, 
                                                                            layer_name=last_layer_name, 
                                                                            layer_map=layer_map, 
                                                                            dataset_class_names=args.dataset_class_names,
                                                                            neuron_descriptions_full_path=args.milan_descriptions_full_path, 
                                                                            api_token_full_path=args.openai_api_token_full_path, 
                                                                            explanation_type=args.explanation_type, 
                                                                            gpt_temp=temperature)
            if not os.path.exists(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation)): # [:-5] is for getting rid of ".JPEG"
                os.makedirs(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation))
            save(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation, "probs.npy"), probabilities)
            with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation, "prompt.txt"), "w") as f:
                f.write(prompt)
            if args.explanation_type == "rigid":
                explanation_file_name = "rigid_explanation.txt"
            elif args.explanation_type == "soft":
                explanation_file_name = "soft_explanation.txt"
            with open(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation, explanation_file_name), "w") as f:
                f.write(explanation)
            if not args.do_not_copy_image and not os.path.isfile(os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], variation, image_name)):
                copy2(full_image_path, os.path.join(args.output_parent_path, args.dataset, args.cnn, image_name[:-5], image_name))
            


if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./experiment_2_2_resutls")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main(args)