import os
import torch
import numpy as np
import torch.nn
import sys
from modeci_mdf.interfaces.pytorch import importer


# Load Model
script_dir = os.path.dirname(sys.argv[0])
mdf_name = "mlp_classifier.json"

file_path = os.path.join(os.getcwd(), script_dir, mdf_name)

models = importer.mdf_to_pytorch(file_path, eval_models=True)
model = models["mlp_classifier"]
model.eval()

# Iterate on training data, feed forward and log accuracy

imgs_path = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), "example_data", "imgs.npy")
labels_path = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), "example_data","labels.npy")

imgs = np.load(imgs_path)
labels = np.load(labels_path)

matches = 0
for i in range(len(imgs)):
    img = torch.Tensor(imgs[i,:,:]).view(-1, 14*14)
    target = labels[i]
    prediction = model(img)
    match = target==int(prediction)
    if match: matches+=1
    print('Image %i: target: %s, prediction: %s, match: %s'%(i, target, prediction, match))

accuracy = (100.*matches)/len(imgs)
print('Matches: %i/%i, accuracy: %s%%'%(matches,len(imgs), accuracy))
assert accuracy>97
