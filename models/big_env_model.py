import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoencoder import AutoEncoder, AutoEncoder_MLP
from models.env_model import AE_R
from models.autogressiveGRU import PRED_MODEL
from models.vanilla_vae import VanillaVAE

class ENV_MODEL(nn.Module):

    def __init__(self, input_size, num_action, hidden_size, hidden_dims=None):
        super(ENV_MODEL, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.gru = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)

        '''
        models
        '''
        self.connection_recog = AE_R(3, num_action, hidden_size, input_size)
        self.feature_recog = AutoEncoder(1, latent_dim=input_size)
        self.Autogressive_model = PRED_MODEL(input_size=input_size + num_action, num_fram=3, num_action=num_action,
                                             hidden_size=hidden_size, queue_size=5)

    def get_feature_encode(self, state):
        return self.feature_recog.encode(state)

    def get_z_encode(self, state, bg):
        return self.connection_recog.encode(state, bg)

    def pred_n_step(self, step, z, a, background):
        x = torch.cat(z, a)

        return self.Autogressive_model.predict(step, x, background)

    def loss_function(self, rec_feature, feature, org_hat, org, x_p, a_p, x_p_hat, a_p_hat):
        recons_feature_loss = F.mse_loss(rec_feature, feature)
        recons_c_org_loss = F.mse_loss(org_hat, org)
        cros_loss = nn.CrossEntropyLoss()
        pred_loss_z = F.mse_loss(x_p_hat, x_p)
        # print(a_p_hat[0][0], a_p[0][0])
        pred_loss_a = cros_loss(a_p_hat, a_p)
        pred_loss_a_ = F.mse_loss(a_p_hat, a_p)
        pred_a = pred_loss_a + pred_loss_a_
        return recons_feature_loss + recons_c_org_loss + pred_loss_z + 0.5 * pred_a, recons_feature_loss, recons_c_org_loss, pred_loss_z, pred_a

    def forward(self, background, feature, org, action):
        # print(feature.shape)
        latent_feature = self.feature_recog.encode(feature)
        # print(latent_feature.shape)
        rec_feature = self.feature_recog.decode(latent_feature)
        # print(rec_feature.shape)
        print(latent_feature.shape)
        c_feature = torch.transpose(latent_feature.unfold(0, 3, 1), 1, 2)
        background = background[2:]
        org = org[2:]
        # print(c_feature.shape)
        # print(background.shape)
        # print(org.shape)
        z = self.connection_recog.encode(c_feature, background)
        # print(z.shape)
        org_hat = self.connection_recog.decode(z)

        action = action[2:]
        # print(org_hat.shape)
        # print(action.shape)
        x = torch.cat((z, action), dim=1)
        # print(x.shape)
        x_p = x[1:]
        # print(x_p.shape)
        x_p = torch.transpose(x_p.unfold(0, 10, 1), 1, 2)
        # print('dasf')
        x = x[:-1]
        x = torch.transpose(x.unfold(0, 10, 1), 1, 2)
        # print(x.shape)
        x_p_hat, a_p_hat = self.Autogressive_model(x, background[10:])
        xp = x_p[:, :, :x_p_hat.shape[2]]
        # print(xp.shape)
        ap = x_p[:, :, x_p_hat.shape[2]:]
        return rec_feature, feature, org_hat, org, xp, ap, x_p_hat, a_p_hat



class ENV_MODEL_V2(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_dims=None):
        super(ENV_MODEL_V2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        # self.gru = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)

        '''
        models
        '''
        self.bce = nn.BCELoss()
        self.connection_recog = AE_R(3,  hidden_size, input_size)
        # self.feature_recog = AutoEncoder(1, latent_dim=input_size)
        self.feature_recog = VanillaVAE(in_channels=1, latent_dim=input_size)
        # self.Autogressive_model = PRED_MODEL(input_size=input_size + num_action, num_fram=3, num_action=num_action,
        #                                      hidden_size=hidden_size, queue_size=5)

    def get_feature_encode_train(self, state):
        t1 = state[:, 0, :, :]  # 第一个时间步
        t2 = state[:, 1, :, :]  # 第二个时间步
        t3 = state[:, 2, :, :]  # 第三个时间步
        t4 = state[:, 3, :, :]  # 第四个时间步
        mu, log_var = self.feature_recog.encode(t1)

        a1 = self.feature_recog.reparameterize(mu, log_var)

        mu, log_var = self.feature_recog.encode(t2)

        a2 = self.feature_recog.reparameterize(mu, log_var)

        mu, log_var = self.feature_recog.encode(t3)

        a3 = self.feature_recog.reparameterize(mu, log_var)

        mu, log_var = self.feature_recog.encode(t4)

        a4 = self.feature_recog.reparameterize(mu, log_var)
        # a1 = self.feature_recog.encode(t1)
        # a2 = self.feature_recog.encode(t2)
        # a3 = self.feature_recog.encode(t3)
        # a4 = self.feature_recog.encode(t4)
        result = torch.stack((a1, a2, a3, a4), dim=1)  # 沿着第二个维度拼接
        return result
    def get_feature_encode(self, state):
        mu, log_var = self.feature_recog.encode(state)

        a1 = self.feature_recog.reparameterize(mu, log_var)

        # return self.feature_recog.encode(state)
        return a1

    def get_z_encode(self, state, bg):
        return self.connection_recog.encode(state, bg)

    def loss_function(self, rec_feature, feature, org_hat, org):
        recons_feature_loss = self.bce(rec_feature, feature)
        # print(org_hat.shape, org.shape)
        recons_c_org_loss = self.bce(org_hat, org[:, 0, :, :])

        return recons_feature_loss + recons_c_org_loss, recons_feature_loss, recons_c_org_loss
    def forward(self, background, feature, org):
        # print(feature.shape)
        latent_feature = self.get_feature_encode_train(feature)
        # print(latent_feature.shape)
        rec_feature = self.feature_recog.decode(latent_feature[:, -1, :])
        # print(rec_feature.shape)
        # c_feature = torch.transpose(latent_feature.unfold(0, 3, 1), 1, 2)
        # background = background[2:]
        # org = org[2:]
        # print(c_feature.shape)
        # print(background.shape)
        # print(org.shape)
        z = self.connection_recog.encode(latent_feature, background)
        # print(z.shape)
        org_hat = self.connection_recog.decode(z)


        return rec_feature, feature[:,-1,:,:], org_hat, org














if __name__ == "__main__":
    import torch

    # 构造虚构数据
    batch_size = 64
    sequence_length = 10
    feature_dim = 64
    num_action = 5

    background = torch.randn((batch_size, 60, 45)).to('cuda')  # 3 channels, 64x64 resolution
    feature = torch.randn((batch_size, 1, 210, 160)).to('cuda')
    org = torch.randn((batch_size, 1, 210, 160)).to('cuda')
    action = torch.randint(0, num_action, (batch_size, 5)).to('cuda')
    print(action.shape)
    # 初始化模型
    env_model = ENV_MODEL(input_size=feature_dim, num_action=num_action, hidden_size=128)

    # 将模型移至 CPU 或 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_model.to(device)

    # 前向传播
    rec_feature, feature, org_hat, org, x_p, x_p_hat = env_model(background, feature, org, action)

    loss1, l2, l3 = env_model.loss_function(rec_feature, feature, org_hat, org, x_p, x_p_hat)
    print(loss1, l2, l3)
    # 打印输出
    print("rec_feature shape:", rec_feature.shape)
    print("feature shape:", feature.shape)
    print("org_hat shape:", org_hat.shape)
    print("org shape:", org.shape)
    print("x_p shape:", x_p.shape)
    print("x_p_hat shape:", x_p_hat.shape)




