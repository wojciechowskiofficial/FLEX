from torch import Tensor

def classify(filename: str, model, top_k=5) -> Tensor:
    
    from PIL import Image
    from torchvision import transforms
    import os
    import torch
    import numpy as np
    
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Conversion to 3 channels
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Read the categories
    with open('imagenet_classes.txt', 'r') as f:
        categories = np.asarray([s.strip() for s in f.readlines()])
    # Show top categories per image
    top_prob, top_catid = torch.topk(probabilities, top_k)
    return probabilities, top_prob, categories[top_catid], input_batch, input_tensor

def wikipedify(category):
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia('en')
    mapping = {'lakeside' : 'lake'}
    if category not in mapping.keys():
        page = wiki.page(category)
    else:
        page = wiki.page(mapping[category])
    wiki_text = page.text
    return wiki_text

def aggregate_areas(positions):
    all_positions = ["top-left corner", "top", "top-right corner", 
                     "left", "center", "right", 
                     "bottom-left corner", "bottom", "bottom-right corner"]
    area_dict = {i: all_positions[i] for i in range(9)}
    
    pos_keys = [key for key, value in area_dict.items() if value in positions]
    set_pos_keys = set(pos_keys)

    aggregated_names = []
    used_positions_keys = set()

    areas = {
        "entire top": {0, 1, 2},
        "entire bottom": {6, 7, 8},
        "entire left": {0, 3, 6},
        "entire right": {2, 5, 8},
        "perimeter": {0, 1, 2, 5, 8, 7, 6, 3},
        "center cross": {1, 3, 4, 5, 7},
        "upper half": {0, 1, 2, 3, 4, 5},
        "lower half": {3, 4, 5, 6, 7, 8},
        "left half": {0, 1, 3, 4, 6, 7},
        "right half": {1, 2, 4, 5, 7, 8}
    }

    # if all tiles are active, return "entire image"
    if len(set_pos_keys) >= 7:
        return ["entire image"], []

    for area, keys in areas.items():
        if keys.issubset(set_pos_keys):
            aggregated_names.append(area)
            used_positions_keys.update(keys)

    # remove entire if half is present
    half_entire = [("upper half", "entire top"), ("lower half", "entire bottom"), 
                  ("left half", "entire left"), ("right half", "entire right")]
    for half, entire in half_entire:
        if half in aggregated_names and entire in aggregated_names: aggregated_names.remove(entire)

    # find unused positions
    unused_positions_keys = set_pos_keys - used_positions_keys
    unused_positions = [area_dict[key] for key in unused_positions_keys]
    
    return aggregated_names, unused_positions
