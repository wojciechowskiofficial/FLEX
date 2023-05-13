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