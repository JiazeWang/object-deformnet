from lib.network_t4_vis import DeformNet
import numpy as np
from torchviz import make_dot
import torch

model = DeformNet(6, 1024)
choose = np.load("choose.npy")
img = np.load("img.npy")
points = np.load("points.npy")
cat_id = np.load("cat_id.npy")
prior = np.load("prior.npy")

choose = torch.from_numpy(choose)
img = torch.from_numpy(img)
prior = torch.from_numpy(prior)
cat_id = torch.from_numpy(cat_id)
points = torch.from_numpy(points)
y, y1 =model(points, img, choose, cat_id, prior)

g = make_dot(y)
g.render('y', view=False)

g = make_dot(y1)
g.render('y1', view=False)
