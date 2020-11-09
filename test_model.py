import torch

from transformer1 import DeformNet
estimator = DeformNet(6, opt.1024)
estimator.cuda()
estimator = nn.DataParallel(estimator)
estimator.load_state_dict(torch.load("transformer1.pth"))
print("success")

from transformer2 import DeformNet
estimator = DeformNet(6, opt.1024)
estimator.cuda()
estimator = nn.DataParallel(estimator)
estimator.load_state_dict(torch.load("transformer2.pth"))
print("success")

from transfromer2_recurrent3.py import DeformNet
estimator = DeformNet(6, opt.1024)
estimator.cuda()
estimator = nn.DataParallel(estimator)
estimator.load_state_dict(torch.load("transfromer2_recurrent3.pth"))
print("success")
