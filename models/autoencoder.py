import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.types_ import *



class AutoEncoder_210(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AutoEncoder_210, self).__init__()

        self.latent_dim = latent_dim
        self.in_channel = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                          kernel_size=8, stride=4, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[0], out_channels=hidden_dims[1],
                          kernel_size=6, stride=3, padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[1], out_channels=hidden_dims[2],
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[2], out_channels=hidden_dims[3],
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.feature = nn.Linear(3072, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 3072)

        hidden_dims.reverse()

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0],
                                   hidden_dims[1],
                                   kernel_size=8,
                                   stride=4,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[1],
                                   hidden_dims[2],
                                   kernel_size=4,
                                   stride=3,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[2],
                                   hidden_dims[3],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channel,
                      kernel_size=3, padding=(1, 0)),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(input)
        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        feature = self.feature(result)

        return feature

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 3)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        feature = self.encode(input)

        return [self.decode(feature), input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]


        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)


        loss = recons_loss

        return {'loss': loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the models
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.types_ import *

import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.types_ import *



class AutoEncoder_210(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AutoEncoder_210, self).__init__()

        self.latent_dim = latent_dim
        self.in_channel = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                          kernel_size=8, stride=4, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[0], out_channels=hidden_dims[1],
                          kernel_size=6, stride=3, padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[1], out_channels=hidden_dims[2],
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[2], out_channels=hidden_dims[3],
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.feature = nn.Linear(3072, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 3072)

        hidden_dims.reverse()

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0],
                                   hidden_dims[1],
                                   kernel_size=8,
                                   stride=4,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[1],
                                   hidden_dims[2],
                                   kernel_size=4,
                                   stride=3,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[2],
                                   hidden_dims[3],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channel,
                      kernel_size=3, padding=(1, 0)),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(input)
        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        feature = self.feature(result)

        return feature

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 4, 3)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        feature = self.encode(input)

        return [self.decode(feature), input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]


        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)


        loss = recons_loss

        return {'loss': loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the models
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from models.types_ import *



class AutoEncoder_MLP(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AutoEncoder_MLP, self).__init__()

        self.latent_dim = latent_dim
        self.in_channel = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels=hidden_dims[0],
        #                   kernel_size=8, stride=4, padding=1),
        #         nn.BatchNorm2d(hidden_dims[0]),
        #         nn.LeakyReLU())
        # )
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(hidden_dims[0], out_channels=hidden_dims[1],
        #                   kernel_size=6, stride=3, padding=1),
        #         nn.BatchNorm2d(hidden_dims[1]),
        #         nn.LeakyReLU())
        # )
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(hidden_dims[1], out_channels=hidden_dims[2],
        #                   kernel_size=4, stride=2, padding=1),
        #         nn.BatchNorm2d(hidden_dims[2]),
        #         nn.LeakyReLU())
        # )
        # modules.append(
        #     nn.Sequential(
        #         nn.Conv2d(hidden_dims[2], out_channels=hidden_dims[3],
        #                   kernel_size=3, stride=2, padding=1),
        #         nn.BatchNorm2d(hidden_dims[3]),
        #         nn.LeakyReLU())
        # )
        self.flatten = nn.Flatten()
        in_features = 84*84  # 计算展平后的大小
        layers = []
        layers.append(nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.LeakyReLU()
        ))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LeakyReLU()
            ))
        # self.mlp = nn.Sequential(*layers)
        self.encoder = nn.Sequential(*layers)

        self.feature = nn.Linear(1024, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 1024)

        hidden_dims.reverse()

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0],
                                   hidden_dims[1],
                                   kernel_size=7,
                                   stride=3,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[1],
                                   hidden_dims[2],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[2],
                                   hidden_dims[3],
                                   kernel_size=3,
                                   stride=3,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channel,
                      kernel_size=3),
            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(input.shape)
        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        feature = self.feature(result)

        return feature

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        print(input.shape)
        feature = self.encode(input)

        return [self.decode(feature), input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]


        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)


        loss = recons_loss

        return {'loss': loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the models
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 84, 84)  # 1样本，3通道，64x64大小的图像

    # 创建一个VanillaVAE模型实例
    latent_dim = 64  # 假设latent_dim为64
    vae = AutoEncoder(in_channels=1, latent_dim=latent_dim)

    # 使用输入张量进行前向传播
    output = vae(input_tensor)

    # 输出是一个包含多个元素的列表，根据需要访问不同的元素
    reconstructed_image = output[0][0]  # 重建的图像
    original_image = input_tensor  # 原始图像

    # 打印形状
    print("原始图像形状:", original_image.shape)
    print("重建图像形状:", reconstructed_image.shape)




# if __name__ == '__main__':
#     input_tensor = torch.randn(1, 1, 210, 160)  # 1样本，3通道，64x64大小的图像
#
#     # 创建一个VanillaVAE模型实例
#     latent_dim = 64  # 假设latent_dim为64
#     vae = AutoEncoder(in_channels=1, latent_dim=latent_dim)
#
#     # 使用输入张量进行前向传播
#     output = vae(input_tensor)
#
#     # 输出是一个包含多个元素的列表，根据需要访问不同的元素
#     reconstructed_image = output[0][0]  # 重建的图像
#     original_image = input_tensor  # 原始图像
#
#     # 打印形状
#     print("原始图像形状:", original_image.shape)
#     print("重建图像形状:", reconstructed_image.shape)

class AutoEncoder(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.in_channel = in_channels
        self.sigmoid = nn.Sigmoid()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels=hidden_dims[0],
                          kernel_size=8, stride=4, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[0], out_channels=hidden_dims[1],
                          kernel_size=6, stride=3, padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[1], out_channels=hidden_dims[2],
                          kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[2], out_channels=hidden_dims[3],
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.feature = nn.Linear(1024, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 1024)

        hidden_dims.reverse()

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[0],
                                   hidden_dims[1],
                                   kernel_size=7,
                                   stride=3,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[1],
                                   hidden_dims[2],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[2]),
                nn.LeakyReLU())
        )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[2],
                                   hidden_dims[3],
                                   kernel_size=3,
                                   stride=3,
                                   padding=1),
                nn.BatchNorm2d(hidden_dims[3]),
                nn.LeakyReLU())
        )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channel,
                      kernel_size=3),
            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # print(input.shape)
        result = self.encoder(input)
        # print(result.shape)
        result = torch.flatten(result, start_dim=1)
        feature = self.feature(result)

        return self.sigmoid(feature)

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        feature = self.encode(input)

        return [self.decode(feature), input]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]


        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)


        loss = recons_loss

        return {'loss': loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the models
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



if __name__ == '__main__':
    input_tensor = torch.randn(1, 1, 84, 84)  # 1样本，3通道，64x64大小的图像

    # 创建一个VanillaVAE模型实例
    latent_dim = 64  # 假设latent_dim为64
    vae = AutoEncoder(in_channels=1, latent_dim=latent_dim)

    # 使用输入张量进行前向传播
    output = vae(input_tensor)

    # 输出是一个包含多个元素的列表，根据需要访问不同的元素
    reconstructed_image = output[0][0]  # 重建的图像
    original_image = input_tensor  # 原始图像

    # 打印形状
    print("原始图像形状:", original_image.shape)
    print("重建图像形状:", reconstructed_image.shape)




# if __name__ == '__main__':
#     input_tensor = torch.randn(1, 1, 210, 160)  # 1样本，3通道，64x64大小的图像
#
#     # 创建一个VanillaVAE模型实例
#     latent_dim = 64  # 假设latent_dim为64
#     vae = AutoEncoder(in_channels=1, latent_dim=latent_dim)
#
#     # 使用输入张量进行前向传播
#     output = vae(input_tensor)
#
#     # 输出是一个包含多个元素的列表，根据需要访问不同的元素
#     reconstructed_image = output[0][0]  # 重建的图像
#     original_image = input_tensor  # 原始图像
#
#     # 打印形状
#     print("原始图像形状:", original_image.shape)
#     print("重建图像形状:", reconstructed_image.shape)