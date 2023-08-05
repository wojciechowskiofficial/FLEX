from male import run_pipeline_single_decision
import os
import pandas as pd
from tqdm import tqdm
from numpy import save
from shutil import copy2


VAL_SET_PARENT_PATH = "./val"
VAL_IMAGES_NAMES_FULL_PATH = "./val_paths_experiments_0.csv" # filtered by class
MILAN_DESCRIPTIONS_FULL_PATH = "./milan_results/resnet18_imagenet.csv"
OPENAI_API_TOKEN_FULL_PATH = "/home/adamwsl/.gpt_api_token/token.txt"
CNN = "resnet18"
DATASET = "imagenet"
OUTPUT_PARENT_PATH = "./basic_explanations"
EXPLANATION_TYPE = "rigid"
COPY_IMAGE = True


def Main():
    # Prepare for processing 
    if CNN == "resnet18":
        from torchvision.models import resnet18
        last_layer_name = "layer4"
        model = resnet18(pretrained=True)
        _ = model.eval()
        
        layer_map = {'conv1' : model.bn1, 
                    'layer1' : model.layer1[1].bn2, 
                    'layer2' : model.layer2[1].bn2, 
                    'layer3' : model.layer3[1].bn2, 
                    'layer4' : model.layer4[1].bn2}
    elif CNN == "alexnet":
        from torchvision.models import alexnet
        last_layer_name = "conv5"
        model = alexnet(pretrained=True)
        _ = model.eval()
        
        layer_map = {'conv1' : model.features[0], 
                     'conv2' : model.features[3], 
                     'conv3' : model.features[6], 
                     'conv4' : model.features[8], 
                     'conv5' : model.features[10]}
    
    df = pd.read_csv(VAL_IMAGES_NAMES_FULL_PATH, header=None)
    image_names = df.iloc[:,0].tolist()
    
    # Explain
    for image_name in tqdm(image_names):
        full_image_path = os.path.join(VAL_SET_PARENT_PATH, image_name)
        
        probabilities, prompt, explanation = run_pipeline_single_decision(model=model, 
                                                                          full_image_path=full_image_path, 
                                                                          layer_name=last_layer_name, 
                                                                          layer_map=layer_map, 
                                                                          neuron_descriptions_full_path=MILAN_DESCRIPTIONS_FULL_PATH, 
                                                                          api_token_full_path=OPENAI_API_TOKEN_FULL_PATH, 
                                                                          explanation_type=EXPLANATION_TYPE)
        if not os.path.exists(os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5])): # [:-5] is for getting rid of ".JPEG"
            os.makedirs(os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5]))
        save(os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5], "probs.npy"), probabilities)
        with open(os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5], "prompt.txt"), "w") as f:
            f.write(prompt)
        if EXPLANATION_TYPE == "rigid":
            explanation_file_name = "rigid_explanation.txt"
        elif EXPLANATION_TYPE == "soft":
            explanation_file_name = "soft_explanation.txt"
        with open(os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5], explanation_file_name), "w") as f:
            f.write(explanation)
        if COPY_IMAGE and not os.path.isfile(os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5], image_name)):
            copy2(full_image_path, os.path.join(OUTPUT_PARENT_PATH, DATASET, CNN, image_name[:-5], image_name))


if __name__ == "__main__":
    Main()
    