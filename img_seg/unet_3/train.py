import os
import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import FaceDataset
from monai.losses import DiceLoss
import numpy as np
from segmentation_models_pytorch import utils

import torchvision.transforms as T



BASE_PATH = 'C:/Users/jun/Downloads/dataset/'
dir_train_img = os.path.join(BASE_PATH, 'train_images')
dir_train_lbl = os.path.join(BASE_PATH, 'train_labels')
dir_val_img = os.path.join(BASE_PATH, 'val_images')
dir_val_lbl = os.path.join(BASE_PATH, 'val_labels')


def main():
    class_names = ['background', 'person']
    class_rgb_values = [[0, 0, 0], [255, 255, 255]]

    select_class_indices = [class_names.index(cls.lower()) for cls in class_names]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    transforms = T.Compose(
        [T.ToTensor()]
    )

    train_dataset = FaceDataset(
        dir_train_img, dir_train_lbl,
        class_rgb_values=select_class_rgb_values,
        transforms=transforms,
    )

    valid_dataset = FaceDataset(
        dir_val_img, dir_val_lbl,
        class_rgb_values=select_class_rgb_values,
        transforms=transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=1)

    TRAINING = True
    EPOCHS = 10
    model = UNet()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = utils.losses.DiceLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.00008),
    ])

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    if TRAINING:

        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, 'C:/Users/jun/Downloads/dataset/model/model.pth')
                print('Model saved!')

if __name__ == '__main__':
    main()