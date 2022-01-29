
import hydra
from omegaconf import DictConfig

from ColorizationDataset import make_dataloaders
from learning import train_model
from logger import Logger
from models import MainModel
from utils import seed_everything, get_paths
from fastai.data.external import untar_data, URLs
import torch


@hydra.main(config_path="train_config", config_name="config_v1")
def train(cfg: DictConfig):
    Logger.prepare(cfg.project_name)

    path = untar_data(URLs.COCO_SAMPLE)
    path = str(path) + "/train_sample"

    seed_everything(cfg.seed)

    train_paths, val_paths = get_paths(path)

    train_dl = make_dataloaders(paths=train_paths, split='train', batch_size=cfg.batch_size)
    val_dl = make_dataloaders(paths=val_paths, split='val', batch_size=cfg.batch_size)

    model = MainModel(lr_G=cfg.lr_G,
                      lr_D=cfg.lr_D,
                      beta1=cfg.beta1,
                      beta2=cfg.beta2,
                      lambda_L1=cfg.lambda_L1)
    with Logger(model=model, name=cfg.run_name) as logger:
        train_model(model, train_dl, val_dl, cfg.epochs, logger)
    torch.save(model.state_dict(), "main_model.pt")

if __name__ == "__main__":
    train()
