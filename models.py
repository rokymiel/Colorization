import torch
from torch import nn, optim
from loss import GANLoss
from utils import init_model


class UnetBlock(nn.Module):

    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        """
        :param nf: число каналов текущего уровня (равно числу каналов выхода  и если не outermost то и входа) / число каналов после up
        :param ni: число каналов на вход внутренниму блоку Unet / число каналов после down
        :param submodule: внутренние блоки
        :param input_c: входные каналы (по умолчанию равно nf)
        :param dropout: нужен ли дропаут
        :param innermost: самый внутренний блок Unet
        :param outermost: Общий блок, содержащий всю сеть
        """
        super().__init__()

        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        """

        :param input_c:  входной канал
        :param output_c: выходной канал
        :param n_down: число слоев вниз
        :param num_filters: размер баового фильтра
        """
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        """
        :param input_c: число входных каналов
        :param num_filters: размер начального фильтра
        :param n_down:
        """
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, in_c, out_c, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        """
        :param in_c: входные каналы
        :param out_c: выходные каналы
        :param k: kernel_size
        :param s: stride
        :param p: padding
        :param norm: добавлять ли нормализацию
        :param act: добавлять ли активацию
        :return:
        """
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(out_c)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100., gan_mode='vanilla'):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode=gan_mode).to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L) # Предсказание

    def validate(self, data):
        """
        :return: возвращает предсказание и оригинальное изображение разбитое на ab и L
        """
        self.net_G.eval()

        with torch.no_grad():
            self.setup_input(data)
            self.forward()
        self.net_G.train()

        return self.fake_color.detach(), self.ab, self.L

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1) # склеиваем в LAB изображение
        fake_preds = self.net_D(fake_image.detach()) # предсказания по сегментам
        self.loss_D_fake = self.GANcriterion(fake_preds, False) # смотрим на то, насколько хорошо дискриминатор может определить, что подали предсказание с фейковыми цветами
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True) # учим дискриминатор выявлять изображение с реальными цветами
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 # в статьях потому что эмперически более стабильно
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1) # склеиваем в LAB изображение
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True) # Мы хотим чтобы  дискриминатор был обманут
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1 # попиксельно заставляет быть схожим
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward() # делаем предсказания раскраски
        # Сначала обучаем дискриминатор
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # После этого обучаем наш генератор
        self.net_G.train()
        # Замораживаем веса дискриминатора, чтобы обучение генераторы не было с ним связано
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()