from src.male import run_pipeline_single_decision
import os
import pandas as pd
from tqdm import tqdm
from numpy import save, argmax, mean, std
from shutil import copy2
from argparse import ArgumentParser
from torch import ones_like


def Main(args):
    # prepare for processing 
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
        
    df = pd.read_csv("masking.tsv", sep="\t")
    
    
    # explain
    counter = 0
    avg, is_class_changed_list = [], []
    top_n_neurons = 5
    for _, row in tqdm(df.iterrows()):
        sat = eval(row["SAT"])
        if type(sat) is not tuple:
            sat = (sat,)
        full_image_path = os.path.join("/home/adamwsl/MALE/various_method_explanations", 
                                       row["image"], 
                                       row["image"] + ".JPEG")
        
        is_class_changed, prob_diff = run_pipeline_single_decision(model=model, 
                                                                   full_image_path=full_image_path, 
                                                                   layer_name=last_layer_name, 
                                                                   layer_map=layer_map, 
                                                                   dataset_class_names=args.dataset_class_names,
                                                                   neuron_descriptions_full_path=args.milan_descriptions_full_path, 
                                                                   api_token_full_path=args.openai_api_token_full_path, 
                                                                   explanation_type=args.explanation_type, 
                                                                   neuron_ids=sat, 
                                                                   top_n_neurons=top_n_neurons)
        avg.append(prob_diff)
        counter += int(is_class_changed)
       
    print("my", counter, mean(avg), std(avg))
       
        
if __name__ == "__main__":
    parser = ArgumentParser(description='Run pipeline on whole validation set')
    parser.add_argument("--val_set_parent_path", default="./val")
    parser.add_argument("--val_images_names_full_path", default="./metadata/val_paths_experiments_0.csv")
    parser.add_argument("--milan_descriptions_full_path", default="./milan_results/resnet18_imagenet.csv")
    parser.add_argument("--openai_api_token_full_path", default="/home/adamwsl/.gpt_api_token/token.txt")
    parser.add_argument("--cnn", default="resnet18")
    parser.add_argument("--dataset", default="imagenet")
    parser.add_argument("--output_parent_path", default="./basic_explanations_gpt4")
    parser.add_argument("--explanation_type", default="soft")
    parser.add_argument("--do_not_copy_image", action="store_true", default=False)
    parser.add_argument("--dataset_class_names", default="./metadata/imagenet_classes.txt")
    args = parser.parse_args()
    
    Main(args)