import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchio import Image
from torchvision.transforms import Compose

transforms_dict = {
    tio.RandomAffine(): 0.55,
    tio.RandomElasticDeformation(): 0.25
}

transforms_dict2 = {
    tio.RandomBlur(): 0.25,
    tio.RandomMotion(): 0.25
}
# for aumentation
transform_flip = tio.OneOf(transforms_dict)


class ADNIDataloaderAllData(Dataset):

    def __init__(self, df, root_dir, transform):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 1])
        subject = tio.Subject(img=Image(img_path, type=tio.INTENSITY))

        if (self.transform):
            # in training phase
            transformations = (
                tio.ZNormalization(),
                tio.Resample(target=2, pre_affine_name="affine"),  # preprocessing
                tio.OneOf(transforms_dict),
                tio.OneOf(transforms_dict2)
            )
        else:
            # in validation and testing phase
            transformations = (
                tio.ZNormalization(),
                tio.Resample(target=2, pre_affine_name="affine")  # preprocessing
            )

        transformations = Compose(transformations)
        transformed_image = transformations(subject)

        get_image = transformed_image.img
        tensor_resampled_image = get_image.data
        tensor_resampled_image = tensor_resampled_image.unsqueeze(dim=0)  # adding batch size

        resampled_image = torch.nn.functional.interpolate(input=tensor_resampled_image, size=(256, 256, 166),
                                                          mode='trilinear')  # trilinear because it had 5D (mini-batch x channels x height x width x depth)
        resampled_image = np.reshape(resampled_image, (1, 256, 256, 166))

        y_label = 0.0 if self.df.iloc[index, 2] == 'AD' else 1.0
        y_label = torch.tensor(y_label, dtype=torch.float)

        return resampled_image, y_label


def train_val_test_loaders(train_file: str = '',
                           val_file: str = '',
                           test_file: str = '',
                           root_dir: str = '',
                           train_batch: int = 4,
                           test_batch: int = 1,
                           train_workers: int = 4,
                           test_workers: int = 1
                           ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # dataset
    train_ds = pd.read_csv(train_file)
    val_ds = pd.read_csv(val_file)
    test_ds = pd.read_csv(test_file)

    print("len of train and val and test")
    print(len(train_ds), len(val_ds), len(test_ds))

    train_dataset = ADNIDataloaderAllData(df=train_ds,
                                          root_dir=root_dir,
                                          transform=True)
    val_dataset = ADNIDataloaderAllData(df=val_ds,
                                        root_dir=root_dir,
                                        transform=False)

    test_dataset = ADNIDataloaderAllData(df=test_ds,
                                         root_dir=root_dir,
                                         transform=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True,
                              num_workers=train_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=test_batch, shuffle=False, num_workers=test_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch, shuffle=False, num_workers=test_workers)

    return train_loader, val_loader, test_loader
