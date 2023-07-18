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