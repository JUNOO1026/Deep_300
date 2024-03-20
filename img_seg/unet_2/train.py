import os
import glob
import torch
import seaborn as sns
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FaceDataset
from loss import DiceLoss
from model import UNet
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
BASE_PATH = 'C:/Users/jun/Downloads/dataset/'


def main():
    dir_train_img = os.path.join(BASE_PATH, 'train_images')
    dir_train_lbl = os.path.join(BASE_PATH, 'train_labels')
    dir_val_img = os.path.join(BASE_PATH, 'val_images')
    dir_val_lbl = os.path.join(BASE_PATH, 'val_labels')

    train_dataset = FaceDataset(
        dir_train_img, dir_train_lbl
    )
    val_dataset = FaceDataset(
        dir_val_img, dir_val_lbl
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    TRAINING = True
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(2)
    loss = utils.losses.DiceLoss()
    metrics = [utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.)
    ])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5
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
            print(f"\nEPOCH: {i}")
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, 'C:/Users/jun/Downloads/dataset/model/model.pth')
                print('Model Saved!')

if __name__ == '__main__':
    main()