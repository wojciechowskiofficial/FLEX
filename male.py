from itertools import product
import numpy as np
import torch
from torch import argmax, argsort, relu, amax
import pandas as pd
from captum.attr import LayerLRP, LayerActivation
from copy import deepcopy
from typing import List, Dict, Tuple
import os
import openai
from utils import *


def get_important_neurons(how_much_highest, 
                          input_batch, 
                          model, 
                          layer_names, 
                          layer_map, 
                          descriptions, 
                          probabilities):

    
    per_layer_results = {layer_name : dict() for layer_name in layer_names}
    per_layer_activations = deepcopy(per_layer_results)

    for layer_name in layer_names:
        attribution_lrp = LayerLRP(model, layer_map[layer_name]).attribute(input_batch, argmax(probabilities))
        attribution_lrp.detach_()
        attribution_lrp = relu(attribution_lrp)

        sorted_ids = argsort(amax(attribution_lrp, dim=(2, 3)), descending=True).squeeze_(0)
        query = descriptions[descriptions['layer'] == layer_name]
        highest_activations_query = query.iloc[sorted_ids][:how_much_highest]    
        
        attribution_activations = LayerActivation(model, layer_map[layer_name]).attribute(input_batch)

        for _, r in highest_activations_query.iterrows():
            name = r['description']
            viz = attribution_activations[0, r['unit'], ...].numpy()
            per_layer_results[layer_name][r['unit']] = name
            per_layer_activations[layer_name][r['unit']] = viz
    
    return per_layer_results, per_layer_activations


def _get_position(matrix):
    # Detect active subsquares 
    active_squares = []
    rows, cols = matrix.shape
    sub_rows, sub_cols = rows // 3, cols // 3

    for i, j in product(range(3), range(3)):
        start_row, end_row = i*sub_rows, (i+1)*sub_rows if i != 2 else rows
        start_col, end_col = j*sub_cols, (j+1)*sub_cols if j != 2 else cols
        square = matrix[start_row:end_row, start_col:end_col]
        if np.any(square == 1):
            active_squares.append(i*3 + j)
            
    # Map to NL
    mapping = {0 : "top-left corner", 
               1 : "top", 
               2 : "top-right corner", 
               3 : "left", 
               4 : "center", 
               5 : "right", 
               6 : "bottom-left corner", 
               7 : "bottom", 
               8 : "bottom-right corner"}
    primary_positions = [mapping[el] for el in active_squares]
    agg_pos, prim_pos = aggregate_areas(primary_positions)

    return agg_pos + prim_pos

def get_positions(per_layer_results, per_layer_activations, viz=False):
    import cv2
    from copy import deepcopy
    from matplotlib import pyplot as plt
    per_layer_results = deepcopy(per_layer_results)
    per_layer_positions = deepcopy(per_layer_activations)
    for ln in per_layer_results.keys():
        for unit_id, words in per_layer_results[ln].items():
            fm = per_layer_activations[ln][unit_id]
            _, thresh = cv2.threshold(fm, np.max(fm) * .5, np.max(fm), 0)
            thresh = cv2.normalize(thresh, None, 0, 1., cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            per_layer_positions[ln][unit_id] = _get_position(thresh)
            if viz:
                plt.colorbar(plt.imshow(thresh, cmap='jet'))
                plt.show()
    return per_layer_positions
            
def _get_activation(model, layer, input_tensor):
    activations = []

    def hook(module, input, output):
        activations.append(output)

    handle = layer.register_forward_hook(hook)
    model(input_tensor)
    handle.remove()

    return activations[0]


def associate_channels(input_batch: torch.Tensor, 
                       prev_layer: torch.nn.modules.Conv2d, 
                       prev_out_channels: int,
                       curr_layer: torch.nn.modules.Conv2d, 
                       target_ch_id: int, 
                       model) -> List[int]:
    with torch.no_grad():
        # Get activations of both layers given an input_batch
        activations_curr = _get_activation(model, curr_layer, input_batch)
        activations_curr = activations_curr[:,target_ch_id,:,:]
        activations_prev = _get_activation(model, prev_layer, input_batch)
        
        # Initialize empty placeholder for contributions
        contributions = torch.empty(size=(prev_out_channels,), 
                                    device=input_batch.device)
        
        # Iterate through the channels in the prev_layer
        for ch_id in range(prev_out_channels):
            # Zero out one channel in prev_layer activation tensor
            zeroed_activations_prev = activations_prev.clone()
            zeroed_activations_prev[:,ch_id,:,:] = 0.
                    
            # Compute activation of curr_layer given zeroed out prev_layer
            from_zeroed_activations_curr = curr_layer.forward(zeroed_activations_prev)
            from_zeroed_activations_curr = from_zeroed_activations_curr[:,target_ch_id,:,:]
            
            # Compute the channel score by computing the absolute value of
            # difference between the original output and zeroed out output
            diff = torch.abs(activations_curr - from_zeroed_activations_curr)
            contributions[ch_id] = torch.sum(diff)
        
    return contributions


def run_pipeline_single_decision(model: torch.nn.Module, 
                                 full_image_path: str, 
                                 layer_name: str,
                                 layer_map: Dict[str, torch.nn.Module], 
                                 neuron_descriptions_full_path: str, 
                                 api_token_full_path: str, 
                                 explanation_type: str,
                                 prompt_dir_path: str = "./prompts",
                                 top_neuron_count: int = 10, 
                                 gpt_temp: int = 0.1) -> Tuple[np.ndarray, str, str]:
    

    # Check if the CNN is in the eval mode
    if model.training:
        _ = model.eval()

    # Classify
    probabilities, _, categories, input_batch, input_tensor = classify(full_image_path, 
                                                                                       model)

    # Resize and center the image
    image_center_resized = np.transpose(input_tensor.numpy(), (1, 2, 0))

    # Find most important neurons 
    descriptions = pd.read_csv(neuron_descriptions_full_path)
    per_layer_results, per_layer_activations = get_important_neurons(how_much_highest=top_neuron_count, 
                                                                    input_batch=input_batch, 
                                                                    model=model, 
                                                                    layer_names=[layer_name], 
                                                                    layer_map=layer_map, 
                                                                    descriptions=descriptions, 
                                                                    probabilities=probabilities)

    # Find neuron spatial whereabouts
    per_layer_positions = get_positions(per_layer_results, 
                                        per_layer_activations, 
                                        viz=False)

    # Construct the prompt
    prompt = f'{str(categories[0])}, '
    desc_and_pos = []
    positions = per_layer_positions[layer_name]
    results = per_layer_results[layer_name]
    for k, v in positions.items():
        if explanation_type == 'rigid':
            desc_and_pos.append({'description' : results[k], 'positions' : v, 'id' : k})
        elif explanation_type == "soft":
            desc_and_pos.append({'description' : results[k], 'positions' : v})
        else:
            raise ValueError("Wrong prompt type!")
    prompt += str(desc_and_pos)
    
    with open(api_token_full_path, 'r') as f:
        token = f.readline().strip()
    if explanation_type == "rigid":
        prompt_file = "rigid_prompt.txt"
    elif explanation_type == "soft":
        prompt_file = "soft_prompt.txt"
    with open(os.path.join(prompt_dir_path, prompt_file), 'r') as f:
        whole_prompt = f.readlines()
        
    whole_prompt = ''.join(whole_prompt)
    full_prompt = f'{whole_prompt}PROMPT: "{prompt}"'

    # Run GPT API
    API_KEY = token
    openai.api_key = API_KEY
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "user", "content": full_prompt},
        ], 
    temperature=gpt_temp
    )
    explanation = response["choices"][0]["message"]["content"]

    return probabilities.numpy(), prompt, explanation
