#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision.models import resnet18
from torchsummary import summary
from utils import classify
import numpy as np
from matplotlib import pyplot as plt
import os
from male import get_important_neurons, get_positions, associate_channels
import openai
from json import loads
from tqdm import tqdm

# In[3]:
with open("val_paths_experiments_0.csv", "r") as f:
    paths = f.readlines()
full_paths, ground_truths = [], []
for line in paths:
    line = line.strip()
    p, l = line.split(",")
    full_paths.append(os.path.join("val", p))
    ground_truths.append(l)

for full_path, ground_truth in tqdm(zip(full_paths, ground_truths)):
    
    filename = full_path

    model = resnet18(pretrained=True)
    _ = model.eval()
    # summary(model, (3, 224, 224))


    # In[5]:

    probabilities, top_probabilities, categories, input_batch, input_tensor = classify(filename, model)


    # In[6]:


    image_center_resized = input_tensor.numpy()
    image_center_resized = np.transpose(image_center_resized, (1, 2, 0))


    # In[7]:


    import pandas as pd
    import os
    descriptions = pd.read_csv(os.path.join('milan_results', 'resnet18_imagenet.csv'))
    #layer_names = ['conv1'] + ['layer' + str(i) for i in range(1, 5, 1)]
    layer_names = ['layer4']
    layer_map = {'conv1' : model.bn1, 
                'layer1' : model.layer1[1].bn2, 
                'layer2' : model.layer2[1].bn2, 
                'layer3' : model.layer3[1].bn2, 
                'layer4' : model.layer4[1].bn2}


    # In[8]:


    per_layer_results, per_layer_activations = get_important_neurons(10, 
                                                                    input_batch, 
                                                                    model, 
                                                                    layer_names, 
                                                                    layer_map, 
                                                                    descriptions, 
                                                                    probabilities)


    # In[9]:


    per_layer_positions = get_positions(per_layer_results, per_layer_activations, viz=True)


    # In[10]:


    prompt = str(categories[0]) + ', '
    tmp = []
    positions = per_layer_positions['layer4']
    results = per_layer_results['layer4']
    for k, v in positions.items():
        tmp.append({'description' : results[k], 'positions' : [], 'id' : k})
        if len(v) <= 3:
            tmp[-1]['positions'] = v
    prompt += str(tmp)


    # In[11]:


    with open('/home/adamwsl/.gpt_api_token/token.txt', 'r') as f:
        token = f.readline().strip()
    with open('prompts/system_prompt.txt', 'r') as f:
        system_prompt = f.readline()
    with open('prompts/example_prompt_1.txt', 'r') as f:
        example_prompt_1 = f.readline()
    with open('prompts/example_output_1.txt', 'r') as f:
        example_output_1 = f.readline()
    full_prompt = 'PROMPT: "' + prompt + '"'


    # In[12]:


    with open('prompts/full_prompt.txt', 'r') as f:
        whole_prompt = f.readlines()
    whole_prompt = ''.join(whole_prompt)
    full_prompt = whole_prompt + 'PROMPT: "' + prompt + '"'


    # In[18]:


    API_KEY = token
    openai.api_key = API_KEY
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "user", "content": full_prompt},
        ], 
    temperature=0.1
    )


    # In[19]:
    from time import sleep 
    filename = filename.split("/")[-1]
    if not os.path.exists(os.path.join('experiment_0_results', "imagenet", "resnet18", filename[:-5])):
        os.makedirs(os.path.join('experiment_0_results', "imagenet", "resnet18", filename[:-5]))
    np.save(os.path.join('experiment_0_results', "imagenet", "resnet18", filename[:-5], "probs.npy"), probabilities.numpy())
    with open(os.path.join('experiment_0_results', "imagenet", "resnet18", filename[:-5], "prompt.txt"), "w") as f:
        f.write(prompt)
    with open(os.path.join('experiment_0_results', "imagenet", "resnet18", filename[:-5], "explanation.txt"), "w") as f:
        f.write(response["choices"][0]["message"]["content"])
    
    
# %%
