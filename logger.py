import wandb
import torch
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import numpy as np


def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def get_imgs_fig(lab, fake, real, epoch):
    """
    Метод для создания фигуры с изображениями эксперимента
    :param lab: изображение LAB
    :param fake: предсказание
    :param real: оригинальное изображение
    :return: фигура
    """
    fig, axes = plt.subplots(3, 1)

    axes[0].imshow(lab, cmap='gray')
    axes[0].axis("off")
    axes[0].set_title('LAB изображение', fontsize=20)

    axes[1].imshow(fake)
    axes[1].axis("off")
    axes[1].set_title('Раскрашенное', fontsize=20)

    axes[2].imshow(real)
    axes[2].axis("off")
    axes[2].set_title('Оригинал', fontsize=20)

    fig.suptitle(f'Эпоха {epoch}', fontsize=25)
    fig.set_size_inches(5, 15)
    plt.close()

    return fig


class Logger:
    project = None

    @staticmethod
    def prepare(project_name):
        """
        Перед использование логгер обязательно нужно запустить данный метод
        :param project_name: название проекта
        """
        wandb.login()
        Logger.project = project_name

    def __init__(self, name, model=None):
        """
        :param name: название текущего эксперимента
        """
        self.name = name
        self.model = model

    def __enter__(self):
        wandb.init(project=self.project, name=self.name)
        if not self.model is None: wandb.watch(self.model)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

    def log_dict(self, dict, epoch):
        """
        Логирует словарь
        """
        wandb.log(dict, step=epoch)

    def log_images(self, figs, epoch):
        """
        Логирует массив фигур как изображения
        :param figs: массив фигур
        :param epoch: номер эпохи
        """
        images = list(map(lambda x: wandb.Image(x), figs))

        wandb.log({"Изображения": images}, step=epoch)
