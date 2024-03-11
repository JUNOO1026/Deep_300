import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import warnings

import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings('ignore')

SEED = 34

DATA_ROOT = "D:/DL_datasets"


# 학습 때 지속해서 랜덤한 값이 등장하지 않게 랜덤 seed를 정함
MANUAL_SEED = 1
random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
BATCH_SIZE = 32
IMAGE_SIZE = (178, 218)
N_WORKERS = 1


dataset = datasets.ImageFolder(root=os.path.join(DATA_ROOT),
                    transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

print(iter(dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", torch.cuda.is_available())

# 데이터 확인
image = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")

plt.title("Training Images")
image = np.transpose(vutils.make_grid(image[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0))
plt.imshow(image)
plt.show()