import torch
import pandas as pd
import albumentations as A
import cv2

from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader

CONFIG = {"seed": 2022,
          "epochs": 10,
          "img_size": 448,
          "model_name": "CSWin",
          'VARIANT': "B",
          "num_classes": 15587,
          "train_batch_size": 8,
          "valid_batch_size": 8,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          }


ROOT_DIR = 'C:/_dataset/happy-whale-and-dolphin/'
TRAIN_DIR = 'C:/_dataset/happy-whale-and-dolphin/train_images/'
TEST_DIR = 'C:/_dataset/happy-whale-and-dolphin/test_images/'

def get_train_file_path(id, TRAIN_DIR):
    return f"{TRAIN_DIR}/{id}"

df = pd.read_csv(f"{ROOT_DIR}/train.csv")
df['file_path'] = df['image'].apply(get_train_file_path)
df.head()
data_size = len(df)

encoder = LabelEncoder()
df['individual_id'] = encoder.fit_transform(df['individual_id'])

skf = StratifiedKFold(n_splits=CONFIG['n_fold'])

for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.individual_id)):
      df.loc[val_ , "kfold"] = fold

class HappyWhaleDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df['individual_id'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(CONFIG['img_size'], CONFIG['img_size']))
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)
#
data_transforms = {
    "train": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.),

    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = HappyWhaleDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = HappyWhaleDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                              num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'],
                              num_workers=0, shuffle=False, pin_memory=True)

    return train_loader, valid_loader
