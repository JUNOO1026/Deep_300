import os
import torch
import numpy as np
from torchvision import transforms as T
from dataset import FaceDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt



BASE_PATH = 'C:/Users/jun/Downloads/dataset/'
dir_train_img = os.path.join(BASE_PATH, 'train_images')
dir_train_lbl = os.path.join(BASE_PATH, 'train_labels')
dir_val_img = os.path.join(BASE_PATH, 'val_images')
dir_val_lbl = os.path.join(BASE_PATH, 'val_labels')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['background', 'person']
class_rgb_values = [[0, 0, 0], [255, 255, 255]]

select_class_indices = [class_names.index(cls.lower()) for cls in class_names]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

if os.path.exists('C:/Users/jun/Downloads/dataset/model/model.pth'):
    b_model = torch.load('C:/Users/jun/Downloads/dataset/model/model.pth', map_location=DEVICE)

else:
    pass


transforms = T.Compose(
        [T.ToTensor()]
    )

test_dataset = FaceDataset(
    dir_val_img, dir_val_lbl,
    class_rgb_values=select_class_rgb_values,
    transforms=transforms,
)

test_dataloader = DataLoader(test_dataset)


image, mask = test_dataset[0]

x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
pred_mask = b_model(x_tensor)
pred_mask = pred_mask.detach().squeeze().cpu().numpy()
pred_mask = np.transpose(pred_mask, (1, 2, 0))


image = image.transpose((1, 2, 0))
a = np.argmax(image)
print(a.shape)
#
# mask = mask.transpose((1, 2, 0))
#
# print(image.shape)
#
# print(mask.shape)
# print(pred_mask.shape)
# #
# visual = np.hstack((image, mask, pred_mask))
# print(visual.shape)
#
# plt.imshow()
# plt.show()



#
# print(image.shape)
# print(mask.shape)