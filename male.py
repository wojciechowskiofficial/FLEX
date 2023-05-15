import numpy as np
import torch


def get_important_neurons(how_much_highest, 
                          input_batch, 
                          model, 
                          layer_names, 
                          layer_map, 
                          descriptions, 
                          probabilities):
    
    from captum.attr import LayerLRP, LayerActivation
    from torch import argmax, argsort, relu, amax
    from copy import deepcopy
    
    per_layer_results = {layer_name : dict() for layer_name in layer_names}
    per_layer_activations = deepcopy(per_layer_results)

    for layer_name in layer_names:
        attribution_lrp = LayerLRP(model, layer_map[layer_name]).attribute(input_batch, argmax(probabilities))
        attribution_lrp.detach_()
        attribution_lrp = relu(attribution_lrp)
        #input_batch.detach()

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


def _get_position(lv, rv, th, bh, thresh):
    positions = list()
    if np.sum(thresh[:,:lv]) > 0: positions.append('left')
    if np.sum(thresh[:,rv:]) > 0: positions.append('right')
    if np.sum(thresh[:th,:]) > 0: positions.append('top')
    if np.sum(thresh[bh:,:]) > 0: positions.append('bottom')
    if np.sum(thresh[th:bh,lv:rv]) > 0: positions.append('center')
    return positions

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
            lv, rv = thresh.shape[0] // 3, thresh.shape[0] // 3 * 2
            th, bh = lv, rv
            per_layer_positions[ln][unit_id] = _get_position(lv, rv, th, bh, thresh)
            if viz:
                print(per_layer_results[ln][unit_id])
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


from typing import List
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