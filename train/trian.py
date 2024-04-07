import numpy
import numpy as np
import torch.nn
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from network_helper import *
from tqdm.auto import tqdm
import configparser
import os
import importlib

fig, ax = plt.subplots()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

#  实例化configParser对象
config = configparser.ConfigParser()
# -read读取ini文件
config.read(r'..\config\conf.ini', encoding='utf-8')

dim = config.getint('config', 'dim')

batch_size = config.getint('train', 'batch_size')
valid_batch = config.getint('train', 'valid_batch')
start_epoch = config.getint('train', 'start_epoch')
save_rate = config.getint('train', 'save_rate')
epochs = config.getint('train', 'epochs')

loss_type = config.get('train', 'loss_type')
optim_name = config.get('train', 'optim_name')
learning_rate = config.getfloat('train', 'learning_rate')

log_file_name = config.get('log', 'log_file_name')

train_s_path = config.get('dataset', 'train_s_path')
train_s_n_path = config.get('dataset', 'train_s_n_path')
valid_s_path = config.get('dataset', 'valid_s_path')
valid_s_n_path = config.get('dataset', 'valid_s_n_path')
dataset_path = config.get('dataset', 'dataset_path')
dataset_name = config.get('dataset', 'dataset_name')

class_path = config.get('model', 'class_path')
model_name = config.get('model', 'model_name')
model_weight_path = config.get('model', 'model_weight_path')

"""----------------------check dir--------------------------------------"""
base_path = '../result/{}'.format(log_file_name)


def check_dir():
    """
    Check if the output file exists, and create it if it doesn't.
    :return:
    """
    if not os.path.exists(base_path):
        os.mkdir(base_path)
        os.mkdir(os.path.join(base_path, 'img'))
        os.mkdir(os.path.join(base_path, 'logs'))
        os.mkdir(os.path.join(base_path, 'weight'))


check_dir()
log_path = os.path.join(
    os.path.join(base_path, "logs"), log_file_name + '.log')
"""---------------------------------Load Dataset-------------------------------------------"""


def get_dataset(s_path, s_n_path):
    """

    :return:
    """
    m = importlib.import_module(dataset_path)
    clz = getattr(m, dataset_name)
    return clz(s_path, s_n_path)


train_dataset = get_dataset(s_path=train_s_path, s_n_path=train_s_n_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = get_dataset(s_path=valid_s_path, s_n_path=valid_s_n_path)
valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=False)
"""---------------------------------Defining Models and Optimizers-------------------------------"""


def get_model():
    """

    :return:
    """
    m = importlib.import_module(class_path)
    clz = getattr(m, model_name)
    return clz()


def get_optimizer(model):
    """

    :return:
    """
    m = importlib.import_module("torch.optim")
    clz = getattr(m, optim_name)
    return clz(model.parameters(), lr=learning_rate)


def load_weight():
    """

    :return:
    """
    if model_weight_path is not None and model_weight_path != '':
        print("load weight: {}".format(model_weight_path))
        model.load_state_dict(torch.load(model_weight_path))


model = get_model()
load_weight()
model.to(device)
optimizer = get_optimizer(model)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def train():
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        loss_batch = 0
        with tqdm(total=len(train_dataloader)) as pbar:
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                b = batch[0].shape[0]
                x = batch[1].to(device)
                condition = batch[0].to(device)
                t = torch.randint(0, timesteps, (batch_size,), device=device).long()
                loss = p_losses(model, x, t, loss_type="huber", condition=condition)
                loss_batch += loss.item()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_description("epoch%d:Loss %f" % (epoch, loss_batch / (step + 1)))
                # model.eval()
        test_use_q_sample_as_model_input(epoch)
        test4SNR(epoch)
        f = open(log_path, "a")
        f.write("epoch{},Loss:{}\n".format(epoch, loss_batch / (step + 1)))
        f.close()
        if epoch % save_rate == 0:
            torch.save(model.state_dict(), base_path + '/weight/model-{}.pth'.format(epoch))
        scheduler.step()


def test_use_q_sample_as_model_input(epoch):
    t = torch.as_tensor([timesteps - 1], dtype=torch.int64)
    condition, x_start = valid_dataset.__getitem__(16 + epoch)
    x_axis = np.log([i + 1 for i in range(dim)])
    x_axis_line = [i + 1 for i in range(dim)]
    ax.plot(x_axis_line, 10 ** train_dataset.resume(x_start.squeeze().flatten(0)), linewidth=0.5)
    fig.savefig(os.path.join(base_path, 'img') + "/{}-1-src.svg".format(epoch))
    plt.cla()

    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=condition, t=t, noise=noise)
    ax.plot(x_axis_line, 10 ** train_dataset.resume(x_noisy.squeeze().flatten(0)), linewidth=0.5)

    fig.savefig(os.path.join(base_path, 'img') + "/{}-3-q_sample.svg".format(epoch))
    plt.cla()

    ax.plot(x_axis_line, 10 ** train_dataset.resume(condition.squeeze().flatten(0)), linewidth=0.5)
    fig.savefig(os.path.join(base_path, 'img') + "/{}-2-condition.svg".format(epoch))
    plt.cla()
    imgs = p_sample_loop(model=model, shape=(2, dim),
                         img=torch.cat((x_noisy.unsqueeze(0).to(device), x_noisy.unsqueeze(0).to(device)), dim=0).to(
                             device),
                         condition=torch.cat((condition.unsqueeze(0).to(device), condition.unsqueeze(0).to(device)),
                                             dim=0).to(device))
    img = []
    for ii in imgs:
        img.append(ii[0])
    imgs = img
    pre_x_start = imgs[0].flatten()
    ax.plot(x_axis_line, 10 ** train_dataset.resume(pre_x_start.squeeze()), linewidth=0.5)
    fig.savefig(os.path.join(base_path, 'img') + "/{}-5-p_sample_start.svg".format(epoch))
    plt.cla()

    pre_x_start_2 = imgs[-1].flatten()
    ax.plot(x_axis_line, 10 ** train_dataset.resume(pre_x_start_2.squeeze()), linewidth=0.5)
    # fig.show()
    fig.savefig(os.path.join(base_path, 'img') + "/{}-6-p_sample_end.svg".format(epoch))
    plt.cla()

    ax.loglog(x_axis, np.absolute(10 ** valid_dataset.resume(condition.squeeze().flatten(0))), linewidth=0.5)
    ax.loglog(x_axis, 10 ** valid_dataset.resume(x_start.squeeze().flatten(0)), linewidth=0.5)
    ax.loglog(x_axis, 10 ** valid_dataset.resume(pre_x_start_2.squeeze()), linewidth=0.5)
    fig.savefig(os.path.join(base_path, 'img') + "/{}-7-all_log.svg".format(epoch))
    plt.cla()

    ax.plot(x_axis_line, 10 ** valid_dataset.resume(condition.squeeze().flatten(0)), linewidth=0.5)
    ax.plot(x_axis_line, 10 ** valid_dataset.resume(x_start.squeeze().flatten(0)), linewidth=0.5)
    ax.plot(x_axis_line, 10 ** valid_dataset.resume(pre_x_start_2.squeeze()), linewidth=0.5)
    fig.savefig(os.path.join(base_path, 'img') + "/{}-8-all_line.svg".format(epoch))
    plt.cla()
    li = []
    li.append(list(10 ** valid_dataset.resume(x_start.squeeze().flatten(0).numpy())))
    li.append(list(10 ** valid_dataset.resume(condition.squeeze().flatten(0).numpy())))
    li.append(list(10 * valid_dataset.resume(pre_x_start_2.squeeze())))

    np.savetxt(os.path.join(base_path, 'img') + "/{}-model.txt".format(epoch), np.asarray(li), fmt='%g', delimiter=' ')


def SNR(data_s, data_n):
    index = 0
    if data_n.min() < 0:
        data_s = data_s - 2 * data_n.min()
        data_n = data_n - 2 * data_n.min()
    data_s = data_s / 1e9
    data_n = data_n / 1e9
    data_s = np.log10(data_s)
    data_n = np.log10(data_n)
    s = np.square(data_s)
    s = np.sum(s, axis=1)
    r = np.square(data_n - data_s)
    r = np.sum(r, axis=1)
    return 10 * np.log10(s / r)


def test4SNR(epoch=0):
    """

    :param epoch:
    :return:
    """

    model.eval()
    snr_batch = 0
    snr_ori_batch = 0
    snr_max = -999999
    with tqdm(total=len(valid_dataloader)) as pbar:
        for step, batch in enumerate(valid_dataloader):
            optimizer.zero_grad()
            b = batch[0].shape[0]
            x = batch[1].to(device)
            condition = batch[0].to(device)
            imgs = p_sample_loop(model=model, shape=batch[0].shape,
                                 img=x,
                                 condition=condition)
            signal_power = np.sum((valid_dataset.resume(x.cpu().numpy().squeeze())) ** 2, axis=1)
            noise_power = np.sum((valid_dataset.resume(x.cpu().numpy().squeeze())
                                  - valid_dataset.resume(imgs[-1].squeeze())
                                  ) ** 2,
                                 axis=1)

            snr = 10 * np.log10(signal_power / noise_power)
            snr_batch += snr.sum() / snr.shape[0]
            if snr.max() > snr_max:
                snr_max = snr.max()
            signal_power = np.sum((valid_dataset.resume(x.cpu().numpy().squeeze())) ** 2, axis=1)
            noise_power = np.sum((valid_dataset.resume(x.cpu().numpy().squeeze())
                                  - valid_dataset.resume(condition.cpu().numpy().squeeze())
                                  ) ** 2,
                                 axis=1)

            snr_ori = 10 * np.log10(signal_power / noise_power)
            snr_ori_batch += snr_ori.sum() / snr_ori.shape[0]
            pbar.update(1)
            pbar.set_description("epoch%d:SNR_ORI %f, SNR_AVG %f,  SNR_MAX %f," % (
                epoch, snr_ori_batch / (step + 1), snr_batch / (step + 1), snr_max))  # 设置描述


if __name__ == '__main__':
    train()
    # test4SNR()
