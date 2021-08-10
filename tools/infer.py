import torch
import argparse
import yaml
from torch import Tensor
from pathlib import Path
from torchvision import io
from torchvision import transforms as T

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import __all__
from utils.utils import time_sync


class PTInfer:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['DEVICE'])
        self.labels = __all__[cfg['DATASET']['NAME']]
        self.model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['MODEL_PATH'], len(self.labels), cfg['TEST']['IMAGE_SIZE'][0])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.img_transforms = T.Compose([
            T.Resize(cfg['TEST']['IMAGE_SIZE']),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        # scale to [0.0, 1.0]
        image = image.float()
        image /= 255

        # normalize
        image = self.img_transforms(image)
        
        # add batch dimension
        image = image.unsqueeze(0).to(self.device)
        return image

    def postprocess(self, prob: Tensor) -> str:
        cls_id = torch.argmax(prob)
        return self.labels[cls_id]

    @torch.no_grad()
    def predict(self, image: Tensor) -> str:
        image = self.preprocess(image)
        start = time_sync()
        pred = self.model(image)
        end = time_sync()
        print(f"PyTorch Model Inference Time: {(end-start)*1000:.2f}ms")

        cls_name = self.postprocess(pred)
        return cls_name

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    file_path = Path(cfg['TEST']['FILE'])
    model = PTInfer(cfg)

    if cfg['TEST']['MODE'] == 'image':
        if file_path.is_file():
            image = io.read_image(str(file_path))
            cls_name = model.predict(image)
            print(f"File: {str(file_path)} >>>>> {cls_name.capitalize()}")
        else:
            files = file_path.glob('*jpg')
            for file in files:
                image = io.read_image(str(file))
                cls_name = model.predict(image)
                print(f"File: {str(file)} >>>>> {cls_name.capitalize()}")
    else:
        raise NotImplementedError