from MLPclass import *
from training_data import *
mlp = MLP(train_data, target_xor, 0.2, 5000)
mlp.train()