from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from datasets import get_sampler
from datasets.transforms import get_train_transforms, get_val_transforms

def get_dataloaders(cfg):
    # augmentations
    train_transforms = get_train_transforms(cfg['IMAGE_SIZE'])
    val_transforms = get_val_transforms(cfg['IMAGE_SIZE'])
    # dataset
    train_dataset = CIFAR10(cfg['DATASET']['ROOT'], True, train_transforms)
    val_dataset = CIFAR10(cfg['DATASET']['ROOT'], False, val_transforms)

    # dataset sampler
    train_sampler, val_sampler = get_sampler(cfg['DDP'], train_dataset, val_dataset)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['num_workers'], drop_last=True, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], num_workers=cfg['num_workers'], pin_memory=True, sampler=val_sampler)

    return train_dataloader, val_dataloader