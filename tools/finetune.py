import argparse
import torch
import yaml
import time
import multiprocessing as mp
import cv2
import pandas as pd

from val import evaluate
from pprint import pprint
from tqdm import tqdm
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import sys


sys.path.insert(0, '.')
from datasets.CIFAR import get_sampler
from models import get_model
from utils.utils import fix_seeds, setup_cudnn, setup_ddp, cleanup_ddp
from utils.metrics import accuracy
from utils.schedulers import get_scheduler
from utils.losses import LabelSmoothCrossEntropy, CrossEntropyLoss
from utils.optimizers import get_optimizer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from datasets.HappyWhale import HappyWhale


def main(cfg, gpu, save_dir):
    start = time.time()
    best_top1_acc, best_top5_acc = 0.0, 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_cfg = cfg['TRAIN']
    optim_cfg = cfg['OPTIMIZER']
    epochs = train_cfg['EPOCHS']
    lr = optim_cfg['LR']

    # dataset
    HappyWhaleDataset = HappyWhale().get_DataSet()

    # dataloader
    train_loader, valid_loader, classes, len_of_train =HappyWhaleDataset.prepare_loaders()

    # training model
    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], None, classes, train_cfg['IMAGE_SIZE'][0])
    ## some models pretrained weights have extra keys, so check them in pretrained loading in model construction
    pretrained_dict = torch.load(cfg['MODEL']['PRETRAINED'], map_location='cpu')
    model.load_state_dict(pretrained_dict, strict=False)

    if cfg['MODEL']['FREEZE']:
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad_ = False
                
    model = model.to(device)
    # model2 = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
    # print(model)
    # print(model2)
    if train_cfg['DDP']: model = DDP(model, device_ids=[gpu])

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = LabelSmoothCrossEntropy()
    optimizer = get_optimizer(model, optim_cfg['NAME'], optim_cfg['LR'], optim_cfg['DECAY'])
    scheduler = get_scheduler(cfg['SCHEDULER'], optimizer)
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(save_dir / 'logs')
    iters_per_epoch = len_of_train // train_cfg['BATCH_SIZE']

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

            print(f"Top-1 Accuracy: {top1_acc:>0.1f} Top-5 Accuracy: {top5_acc:>0.1f} best_top1_acc: {best_top1_acc:>0.1f}")
            writer.add_scalar('val/Top1_Acc', top1_acc, epoch)
            writer.add_scalar('val/Top5_Acc', top5_acc, epoch)

            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top5_acc = top5_acc
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{top1_acc}_{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}.pth")
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