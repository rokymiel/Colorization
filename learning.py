from logger import lab_to_rgb, get_imgs_fig
from utils import create_loss_meters, update_losses, AverageMeter, device
from tqdm import tqdm


def validate(model, val_dl, epoch, logger, batch_num):
    figs = []
    for data in val_dl:
        if batch_num <= 0:
            break
        batch_num -= 1
        fake_color, real_color, L = model.validate(data)
        fake_imgs = lab_to_rgb(L, fake_color)
        real_imgs = lab_to_rgb(L, real_color)

        # получаем массив графиков с изображениями эксперимента для логирования
        figs += list(map(lambda x: get_imgs_fig(x[2].cpu(), x[0], x[1], epoch), list(zip(fake_imgs, real_imgs, L[:, 0]))))

    logger.log_images(figs, epoch)


def train_model(model, train_dl, val_dl, epochs, logger, batch_num_to_val):
    for e in range(epochs):
        # обнуляем метрики
        loss_meter_dict = create_loss_meters()

        for data in tqdm(train_dl, desc=f"Training, epoch {e}", leave=False):
            # step
            model.setup_input(data)
            model.optimize()

            # обновляем метрики
            update_losses(model, loss_meter_dict, count=data['L'].size(0))

        logger.log_dict({k: v.avg for k, v in loss_meter_dict.items()}, e)
        validate(model, val_dl, e, logger, batch_num_to_val)


def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    """
    Предобучает генератор
    """
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl, desc=f"Training generator, epoch {e}", leave=False):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))
        print(f"L1 Loss: {loss_meter.avg:.5f}")
