import os
import torch
import numpy as np
import torch.nn
from modeci_mdf.interfaces.pytorch import mdf_to_pytorch


# Load Model
path = os.path.abspath("mlp_classifier.json")
models = mdf_to_pytorch(path, eval_models=True)
model = models["mlp_classifier"]
model.eval()

# Iterate on training data, feed forward and log accuracy
imgs = np.load("example_data/imgs.npy")
labels = np.load("example_data/labels.npy")

matches = 0
for i in range(len(imgs)):
    img = torch.Tensor(imgs[i,:,:]).view(-1, 14*14)
    target = labels[i]
    prediction = model(img)
    match = target==int(prediction)
    if match: matches+=1
    print('Image %i: target: %s, prediction: %s, match: %s'%(i, target, prediction, match))

print('Matches: %i/%i, accuracy: %s%%'%(matches,len(imgs), (100.*matches)/len(imgs)))

