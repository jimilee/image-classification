import argparse
import torch
import yaml
import time
import multiprocessing as mp
from pprint import pprint
from tqdm import tqdm
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import sys
import cv2
import pandas as pd

sys.path.insert(0, '.')
from datasets import get_sampler
from datasets.transforms import get_train_transforms, get_val_transforms
from models import get_model
from utils.utils import fix_seeds, setup_cudnn, setup_ddp, cleanup_ddp
from utils.schedulers import get_scheduler
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from utils.optimizers import get_optimizer
from val import evaluate
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = 'C:/_dataset/happy-whale-and-dolphin/'
TRAIN_DIR = 'C:/_dataset/happy-whale-and-dolphin/train_images/'
TEST_DIR = 'C:/_dataset/happy-whale-and-dolphin/test_images/'

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
          # ArcFace Hyperparameters
          "s": 30.0,
          "m": 0.50,
          "ls_eps": 0.0,
          "easy_margin": False
          }


def get_train_file_path(id):
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

def main(cfg, gpu, save_dir):
    start = time.time()
    best_top1_acc, best_top5_acc = 0.0, 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg = cfg['TRAIN']
    eval_cfg = cfg['EVAL']
    optim_cfg = cfg['OPTIMIZER']
    epochs = train_cfg['EPOCHS']
    lr = optim_cfg['LR']

    # # augmentations
    # train_transforms = get_train_transforms(train_cfg['IMAGE_SIZE'])
    # val_transforms = get_val_transforms(eval_cfg['IMAGE_SIZE'])
    #
    # # dataset
    # train_dataset = CIFAR10(cfg['DATASET']['ROOT'], True, train_transforms)
    # val_dataset = CIFAR10(cfg['DATASET']['ROOT'], False, val_transforms)
    #
    # # dataset sampler
    # train_sampler, val_sampler = get_sampler(train_cfg['DDP'], train_dataset, val_dataset)
    
    # dataloader
    train_loader, valid_loader = prepare_loaders(df, fold=0)

    # training model
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], None, CONFIG['num_classes'], train_cfg['IMAGE_SIZE'][0])
    ## some models pretrained weights have extra keys, so check them in pretrained loading in model construction
    pretrained_dict = torch.load(cfg['MODEL']['PRETRAINED'], map_location='cpu')
    model.load_state_dict(pretrained_dict, strict=False)

    if cfg['MODEL']['FREEZE']:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad_ = False
                
    model = model.to(device)

    if train_cfg['DDP']: model = DDP(model, device_ids=[gpu])

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = LabelSmoothCrossEntropy()
    optimizer = get_optimizer(model, optim_cfg['NAME'], optim_cfg['LR'], optim_cfg['DECAY'])
    scheduler = get_scheduler(cfg['SCHEDULER'], optimizer)
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(save_dir / 'logs')
    iters_per_epoch = data_size // CONFIG['train_batch_size']

    for epoch in range(epochs):
        model.train()
        
        # if train_cfg['DDP']: train_sampler.set_epoch(epoch)
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        
        for iter, (img, lbl) in pbar:
            img = img.to(device)
            lbl = lbl.to(device)
            optimizer.zero_grad()

            with autocast(enabled=train_cfg['AMP']):
                pred = model(img)
                loss = loss_fn(pred, lbl)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() * img.shape[0]

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")

        train_loss /= iter + 1
        writer.add_scalar('train/loss', train_loss, epoch)
        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch+1) % cfg['TRAIN']['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            # evaluate the model
            top1_acc, top5_acc = evaluate(valid_loader, model, device)

            print(f"Top-1 Accuracy: {top1_acc:>0.1f} Top-5 Accuracy: {top5_acc:>0.1f}")
            writer.add_scalar('val/Top1_Acc', top1_acc, epoch)
            writer.add_scalar('val/Top5_Acc', top5_acc, epoch)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.pth")
            print(f"Best Top-1 Accuracy: {best_top1_acc:>0.1f} Best Top-5 Accuracy: {best_top5_acc:>0.5f}")
        
    writer.close()
    pbar.close()
        
    end = time.gmtime(time.time() - start)
    total_time = time.strftime("%H:%M:%S", end)

    table = [
        ['Best Top-1 Accuracy', best_top1_acc],
        ['Best Top-5 Accuracy', best_top5_acc],
        ['Total Training Time', total_time]
    ]

    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    pprint(cfg)
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    fix_seeds(123)
    setup_cudnn()
    gpu = setup_ddp()
    main(cfg, gpu, save_dir)
    cleanup_ddp()