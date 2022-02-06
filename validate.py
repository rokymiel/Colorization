import glob
import os

import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from ColorizationDataset import make_dataloaders
from logger import lab_to_rgb
from models import MainModel
from utils import device, build_res_unet


@hydra.main(config_path="val_config", config_name="config")
def validate(cfg: DictConfig):
    paths = glob.glob(f"{cfg.data_path}/*.jpg")

    val_dl = make_dataloaders(paths=paths, split='val', batch_size=cfg.batch_size)

    model = None

    if cfg.type == 'v1':
        model = MainModel()
        model.load_state_dict(torch.load(cfg.model_weights_path, map_location=device))

    elif cfg.type == 'v2':
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        model = MainModel(net_G=net_G)
        model.load_state_dict(torch.load(cfg.model_weights_path, map_location=device))
    else:
        ValueError('Unknown type')

    i = 0
    for data in tqdm(val_dl, desc=f"Validating", leave=False):
        fake_color, _, L = model.validate(data)
        fake_imgs = lab_to_rgb(L, fake_color)
        for img in fake_imgs:
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGB')
            img.save(os.path.basename(paths[i]))
            i += 1


if __name__ == "__main__":
    validate()
