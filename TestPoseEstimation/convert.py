import numpy as np
att = np.load("attention_save.npy").squeeze()
att = (att[0] + att[1] + att[2] + att[3]) / 4
argsortresult = np.argsort(att, axis = 1)
print(argsortresult)
