import os

import hydra
from omegaconf import DictConfig
from torch import optim, nn

from ColorizationDataset import make_dataloaders
from learning import train_model, pretrain_generator
from logger import Logger
from models import MainModel
from utils import seed_everything, get_paths, build_res_unet, device
from fastai.data.external import untar_data, URLs
import torch


def train_generator(cfg: DictConfig, path):
    seed_everything(cfg.seed)

    train_paths, val_paths = get_paths(path)

    train_dl = make_dataloaders(paths=train_paths, split='train', batch_size=cfg.batch_size)

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    pretrain_generator(net_G, train_dl, opt, criterion, 20)

    torch.save(net_G.state_dict(), "outputs/End_v2/generator.pt")


@hydra.main(config_path="train_config", config_name="config_v2")
def train(cfg: DictConfig):
    path = cfg.get('data_path')
    if path is None:
        path = untar_data(URLs.COCO_SAMPLE)
        path = str(path) + "/train_sample"

    if cfg.execute_pretrain_generator_only:
        train_generator(cfg, path)
        return

    Logger.prepare(cfg.project_name)

    seed_everything(cfg.seed)

    train_paths, val_paths = get_paths(path)

    train_dl = make_dataloaders(paths=train_paths, split='train', batch_size=cfg.batch_size)
    val_dl = make_dataloaders(paths=val_paths, split='val', batch_size=cfg.batch_size)

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load(f"{hydra.utils.get_original_cwd()}/generator.pt", map_location=device))

    model = MainModel(net_G=net_G,
                      lr_G=cfg.lr_G,
                      lr_D=cfg.lr_D,
                      beta1=cfg.beta1,
                      beta2=cfg.beta2,
                      lambda_L1=cfg.lambda_L1,
                      gan_mode=cfg.gan_mode)
    with Logger(model=model, name=cfg.run_name) as logger:
        train_model(model, train_dl, val_dl, cfg.epochs, logger, cfg.batch_num_to_val)
    torch.save(model.state_dict(), "main_model.pt")


if __name__ == "__main__":
    train()
