import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, window_size, device):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.device = device

        self.attn = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, encoder_output, encoder_hidden):
        """
        :param encoder_output: shape is [max_length, 1, hidden_size]
        :param encoder_hidden: shape is [1, 1, hidden_size]
        :return:
        """
        # shape is [max_length, 1, hidden_size]
        # encoder_hidden.shape[1] is the batch_size
        # print(encoder_output.shape, encoder_hidden.shape)
        hidden_temp = torch.zeros(self.window_size, encoder_hidden.shape[1], self.hidden_size, device=self.device)

        hidden_temp[torch.arange(self.window_size)] = encoder_hidden[0]

        # shape is [max_length, hidden_size * 2]
        att_input = torch.cat((encoder_output, hidden_temp), dim=2)

        # shape is [max_length, 1]
        att_weights = nn.functional.softmax(self.attn(att_input), dim=0)

        # shape is [1, hidden_size] and this is the state vector fed to the policy network
        att_applied = torch.bmm(att_weights.permute(1, 2, 0), encoder_output.transpose(0, 1))

        return att_applied




class AE_R(nn.Module):

    def __init__(self, num_fram, hidden_size, input_size, hidden_dims=None):
        super(AE_R, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.gru = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        # self.conv_feature = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv_background = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv_background_2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1)
        self.fc_b = nn.Linear(26*19, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(input_size, 1)
        self.decoder_input = nn.Linear(32, 1024)
        self.linear = nn.Linear(hidden_size, input_size)
        self.atten = AttentionLayer(128,3, 'cuda')
        self.li = nn.Linear(hidden_size, 32)
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]



        # Build Decoder
        modules = []

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
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3),
            nn.Sigmoid())


        # self.fc1 = nn.Linear(input_size, num_action)

    def decode(self, z):
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

    def rocog_loss(self, recog, org):
        return F.mse_loss(recog, org)


    def encode(self, state, background):
        # print(state.shape, background.shape)
        background = F.relu(self.conv_background(background))
        background = F.relu(self.conv_background_2(background))
        background = background.view(-1, 26 * 19)
        hidden_ = self.fc_b(background).unsqueeze(0)
        # print(hidden_.shape, state.shape)
        # print(state.shape)
        hidden_ = self.relu(hidden_)
        out, hidden = self.gru(state, hidden_)
        # cc = out[:, -1, :]
        # print(cc.shape)
        # print(out.shape, hidden.shape)
        return F.relu(self.li(self.relu(out[:, -1, :])))

    def forward(self, state, background, feature=None):
        # print(state.shape)
        background = F.relu(self.conv_background(background))
        background = F.relu(self.conv_background_2(background))
        background = background.view(-1, 26*19)
        hidden_ = self.fc_b(background).unsqueeze(0)
        # print(hidden_.shape, state.shape)
        # print(state.shape, hidden_.shape)
        hidden_ = self.relu(hidden_)
        # print(state.shape)
        out, hidden = self.gru(state, hidden_)
        dd = F.relu(self.li(self.relu(out[:, -1, :])))
        # print(out.shape, hidden.shape)
        # print(out.shape, 'dsfdsaf')
        # ec = self.atten(out.transpose(0, 1), hidden)/

        recog = self.decode(dd)
        # print(recog.shape)

        return recog








