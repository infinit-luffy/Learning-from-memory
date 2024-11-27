import torch
from models.base import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.types_ import *
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        images = torch.from_numpy(self.data[idx]).unsqueeze(0).float().cuda()

        return images

def train_vae(vae, dataset, batch_size, writer, global_step, train_step=100):

    dataset = CustomDataset(dataset)
    print("training vae, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    log_interval = 100

    for epoch in range(train_step):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{train_step}")
        for batch_idx, data in progress_bar:


            output = vae.forward(data)

            loss = vae.loss_function(*output)

            # 保存重建图像到 TensorBoard 日志
            if batch_idx % log_interval == 0:
                writer.add_scalar("Loss", loss['loss'], global_step)
                writer.add_scalar("R_Loss", loss['Reconstruction_Loss'], global_step)
                writer.add_scalar("KL_Loss", loss['KLD'], global_step)
                writer.add_image(" ImagO", output[1][1], global_step)
                writer.add_image("ImagR", output[0][1], global_step)
                writer.add_image("Imag_diff", output[1][1] - output[0][1], global_step)
            loss = loss['loss']
            global_step += 1

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return global_step

def train_ae(ae, dataset, batch_size, writer, train_step=100):

    dataset = CustomDataset(dataset)
    print("training ae, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    global_step = 0
    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_step}")):

            output = ae.forward(data)

            loss = ae.loss_function(*output)

            # 保存重建图像到 TensorBoard 日志
            if batch_idx % log_interval == 0:
                writer.add_scalar("Loss_P", loss['loss'], global_step)
                writer.add_image(" ImagO_B", output[1][1], global_step)
                writer.add_image("ImagR_B", output[0][1], global_step)

            loss = loss['loss']
            global_step += 1


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("finished training ae")
    return

class CustomDataset_P(Dataset):
    def __init__(self, data, actions, action_size, transform=None):
        self.data = data
        self.actions = actions
        self.action_size = action_size
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data[idx])
        images = torch.from_numpy(self.data[idx]).unsqueeze(0).float().cuda()
        actions = torch.eye(self.action_size)[self.actions[idx]].float().cuda()

        return images, actions


def train_ae_P(ae, dataset, actions, action_size, batch_size, writer, train_step=100):

    dataset = CustomDataset_P(dataset, actions, action_size)
    print("training ae, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    global_step = 0
    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, (data, actions) in enumerate(train_loader):

            output = ae.forward(data)

            loss = ae.loss_function(*output, a=actions)
            print(loss)
            # 保存重建图像到 TensorBoard 日志
            tqdm.write(f"Epoch {epoch + 1}/{train_step}, Batch {batch_idx}, Loss: {loss['loss']:.4f}, Loss_R: {loss['R_loss']:.4f}, action_loss: {loss['action_loss']:.4f}")
            if batch_idx % log_interval == 0:
                writer.add_scalar("Loss_P", loss['loss'], global_step)
                writer.add_image(" ImagO_B", output[1][1], global_step)
                writer.add_image("ImagR_R_B", output[0][1], global_step)
                writer.add_image("Imag_diff", output[1][1] - output[0][1], global_step)



            loss = loss['loss']

            global_step += 1


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("finished training ae")
    return

class CustomDataset_R(Dataset):
    def __init__(self, background, state, org, transform=None):
        self.background = background
        self.state = state
        self.org = org
    def __len__(self):
        return len(self.background)

    def __getitem__(self, idx):

        bg = torch.from_numpy(self.background[idx]).squeeze(1).float().cuda()
        states = torch.from_numpy(self.state[idx]).squeeze(1).squeeze(2).float().cuda()
        orgs = torch.from_numpy(self.org[idx]).unsqueeze(0).float().cuda()
        return bg, states, orgs


def train_ae_R(ae, background, state, org, batch_size, writer, train_step=100):

    dataset = CustomDataset_R(background, state, org)
    print("training ae, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    global_step = 0
    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, (background, state, org) in enumerate(train_loader):
            # print(org.shape)
            output = ae.forward(state, background)
            # print(org.shape)

            loss = ae.rocog_loss(output, org)
            print(loss)
            # 保存重建图像到 TensorBoard 日志
            # tqdm.write(f"Epoch {epoch + 1}/{train_step}, Batch {batch_idx}, Loss: {loss:.4f}")
            if batch_idx % log_interval == 0:
                writer.add_scalar("Loss", loss, global_step)
                writer.add_image(" ImagO", org[0], global_step)
                writer.add_image("ImagR", output[0], global_step)
                # writer.add_image("Imag_diff", output[1][1] - output[0][1], global_step)


            global_step += 1


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("finished training ae_R")
    return

class CustomDataset_env(Dataset):
    def __init__(self, feature_latent, background, org, action, action_size, transform=None):
        self.feature_latent = feature_latent
        self.background = background
        self.org = org
        self.action = action
        self.action_size = action_size


    def __len__(self):
        return len(self.action)

    def __getitem__(self, idx):
        # print(self.data[idx])
        images = torch.from_numpy(self.org[idx]).unsqueeze(0).float().cuda()
        background = torch.tensor(self.background[idx]).squeeze(1).float().cuda()
        feature = torch.tensor(self.feature_latent[idx]).squeeze(1).float().cuda()
        actions = torch.eye(self.action_size)[self.action[idx]].float().cuda()

        return images, background, feature, actions

def train_env_model(env_model, feature, org, action, background, action_size, batch_size, writer, train_step=20):


    dataset = CustomDataset_env(feature_latent=feature, background=background, org=org, action=action, action_size=action_size)
    print("training ae, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(env_model.parameters(), lr=0.003)
    global_step = 0
    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, (org, background, feature, action) in enumerate(train_loader):
            try:
            # print(org.shape)
            # print(feature.shape, org.shape, background.shape, action.shape)

                rec_feature, feature, org_hat, org, xp, ap, x_p_hat, a_p_hat = env_model.forward(background, feature, org, action)
                # print(org.shape)

                loss,  recons_feature_loss, recons_c_org_loss, pred_loss_z, pred_a = env_model.loss_function(rec_feature, feature, org_hat, org, xp, ap, x_p_hat, a_p_hat)
                if batch_idx % log_interval == 0:
                    writer.add_scalar("Loss", loss, global_step)
                    writer.add_scalar("rec_f_Loss", recons_feature_loss, global_step)
                    writer.add_scalar("rec_c_Loss", recons_c_org_loss, global_step)
                    writer.add_scalar("pred_z_loss", pred_loss_z, global_step)
                    writer.add_scalar("pred_a_loss", pred_a, global_step)
                    writer.add_image(" ImagO", org[0], global_step)
                    writer.add_image("ImagR", org_hat[0], global_step)
                    writer.add_image(" BImagO", feature[0], global_step)
                    writer.add_image("BImagR", rec_feature[0], global_step)
                    # writer.add_image("Imag_diff", output[1][1] - output[0][1], global_step)


                global_step += 1



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except:

                print(len(background))



    print("finished training ae_R")
    return


class CustomDataset_env_v2(Dataset):
    def __init__(self, feature_latent, background, org, transform=None):
        self.feature_latent = feature_latent
        self.background = background
        self.org = org
        # self.action = action
        # self.action_size = action_size


    def __len__(self):
        return len(self.feature_latent)

    def __getitem__(self, idx):
        # print(self.data[idx])
        images = torch.from_numpy(self.org[idx]).unsqueeze(0).float().cuda()
        background = torch.tensor(self.background[idx]).squeeze(1).float().cuda()
        feature = torch.tensor(self.feature_latent[idx]).squeeze(1).float().cuda()
        # actions = torch.eye(self.action_size)[self.action[idx]].float().cuda()

        return images, background, feature


def train_env_model_V2(env_model, feature, org, background, batch_size, writer, global_step, train_step=20):


    dataset = CustomDataset_env_v2(feature_latent=feature, background=background, org=org)
    print("training env_model, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(env_model.parameters(), lr=0.0003)

    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, (org, background, feature) in enumerate(train_loader):
            try:
            # print(org.shape)
            # print(feature.shape, org.shape, background.shape)

                rec_feature, feature, org_hat, org = env_model.forward(background, feature, org)
                # print(rec_feature.shape, feature.shape, org_hat.shape, org.shape)
                # print(org_hat.shape)
                # print(org.shape)

                loss,  recons_feature_loss, recons_c_org_loss = env_model.loss_function(rec_feature, feature, org_hat, org)
                if batch_idx % log_interval == 0:
                    writer.add_scalar("Loss", loss, global_step)
                    writer.add_scalar("rec_f_Loss", recons_feature_loss, global_step)
                    writer.add_scalar("rec_c_Loss", recons_c_org_loss, global_step)
                    writer.add_image(" ImagO", org[0], global_step)
                    writer.add_image("ImagR", org_hat[0], global_step)
                    writer.add_image(" BImagO", feature[0], global_step)
                    writer.add_image("BImagR", rec_feature[0], global_step)



                global_step += 1



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except:

                print(len(background))


    return global_step




class CustomDataset_GRUPRED(Dataset):
    def __init__(self, feature_latent, background, connection, action, action_size, transform=None):
        self.feature_latent = feature_latent
        self.background = background
        self.connections = connection
        self.actions = action
        self.action_size = action_size


    def __len__(self):
        return len(self.background)

    def __getitem__(self, idx):
        # print(self.data[idx])
        background = torch.tensor(self.background[idx]).float().cuda()
        feature = torch.tensor(self.feature_latent[idx]).float().cuda()
        connection = torch.tensor(self.connections[idx]).float().cuda()
        actions = torch.tensor(self.actions[idx]).float().cuda()
        # actions = torch.eye(self.action_size)[self.actions[idx]].unsqueeze(0).float().cuda()

        # print(feature.shape, connection.shape, actions.shape, '=======================')
        data = torch.cat((feature, connection, actions), dim=1)


        # print(x.shape)

        # data_target = data[1:]
        # data = data[:-1]
        return data, background
        # x_p[:, :, :x_p.shape[2] - self.action_size], x_p[:, :, x_p.shape[2] - self.action_size:]





def train_GRU_PRED(pred_model, feature, background, connection, action, batch_size, writer, global_step , action_size, train_step=20):



    dataset = CustomDataset_GRUPRED(feature, background, connection, action, action_size=action_size)
    print("training gru, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.0003)
    # global_step = 0
    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, (data, background) in enumerate(train_loader):
            try:
            # print(org.shape)
            # print(feature.shape, org.shape, background.shape, action.shape)
                background = background[10:].squeeze(0)
                x_p = data[1:]
                # print(x_p.shape, 'x_p_shape')
                x_p = torch.transpose(x_p.unfold(0, 10, 1).squeeze(1), 1,2)
                # print('dasf')
                # print(x_p.shape, 'x_p_shape_2')
                x = data[:-1]
                x = torch.transpose(x.unfold(0, 10, 1).squeeze(1), 1, 2)



                x_p_a = x_p[:, :, x_p.shape[2] - action_size:]
                x_p = x_p[:, :, :x_p.shape[2] - action_size]

                # print(x.shape, background.shape, 'dsfa')

                pred_f, a_hat = pred_model.forward(x, background.squeeze(1))
                # print(org.shape)
                # print(pred_f.shape, a_hat.shape, x_p.shape, x_p_a.shape)
                loss,  fc_loss, a_loss_c, a_loss_m = pred_model.loss_function(pred_f, x_p, a_hat, x_p_a)
                if batch_idx % log_interval == 0:
                    writer.add_scalar("Loss", loss, global_step)
                    writer.add_scalar("feature_Loss", fc_loss, global_step)
                    writer.add_scalar("a_Loss_c", a_loss_c, global_step)
                    writer.add_scalar("a_Loss_m", a_loss_m, global_step)



                global_step += 1



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except:

                print(len(background))


    return global_step


def train_GRU_PRED_2(pred_model, feature, background, connection, action, batch_size, writer, global_step , action_size, train_step=20):



    dataset = CustomDataset_GRUPRED(feature, background, connection, action, action_size=action_size)
    print("training gru, dataset_length=", len(dataset))
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(pred_model.parameters(), lr=0.0001)
    # global_step = 0
    log_interval = 100

    for epoch in range(train_step):
        for batch_idx, (data, background) in enumerate(train_loader):
            try:
            # print(org.shape)
            # print(feature.shape, org.shape, background.shape, action.shape)
                background = background[10:].squeeze(0)
                x_p = data[1:]
                # print(x_p.shape, 'x_p_shape')
                x_p = torch.transpose(x_p.unfold(0, 10, 1).squeeze(1), 1,2)
                # print('dasf')
                # print(x_p.shape, 'x_p_shape_2')
                x = data[:-1]
                x = torch.transpose(x.unfold(0, 10, 1).squeeze(1), 1, 2)



                # x_p_a = x_p[:, :, x_p.shape[2] - action_size:]
                # x_p = x_p[:, :, :x_p.shape[2] - action_size]

                # print(x.shape, background.shape, 'dsfa')

                pred_f, a_hat, output_seq = pred_model.forward(x, background.squeeze(1))
                # print(org.shape)
                # print(pred_f.shape, a_hat.shape, x_p.shape, x_p_a.shape)
                loss = pred_model.loss_function_2(output_seq, x)

                if batch_idx % log_interval == 0:
                    writer.add_scalar("Loss", loss, global_step)

                global_step += 1



                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except:

                print(len(background))


    return global_step


