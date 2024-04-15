import logging
import os
import os.path
import time
import numpy as np
import _pickle as pickle
import math
from data_parallel_my_v2 import BalancedDataParallel
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import  StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, roc_auc_score
from torch.nn import functional as F
import pdb
from tools import MultiLabelDataset
import dense_git
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from CAWSNet import SwinTransformer_CAWS_t



logger = logging.getLogger()
logger.setLevel(logging.INFO)

rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname("/root/autodl-tmp/chest-EfficientNet/Logs/")
if not os.path.exists(log_path):
    os.makedirs(log_path)

log_name = log_path + '/' + rq + '.log'  # 设置日志文件名
print(log_name)
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')  # 将日志信息输出到磁盘文件上
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

logger.addHandler(fh)
logger_1 = logging.getLogger()
logger_1.setLevel(logging.INFO)
rq_1 = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))  # time strftime()
log_path_1 = os.path.dirname("/root/autodl-tmp/chest-EfficientNet/Pngs/")
if not os.path.exists(log_path):
    os.makedirs(log_path)

log_name_1 = log_path_1 + '/' + rq_1 + '.png'  # 设置日志文件名
print(log_name_1)
logfile_1 = log_name_1
fh_1 = logging.FileHandler(logfile_1, mode='w')  # 将日志信息输出到磁盘文件上
fh_1.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
logger_1.addHandler(fh_1)

# todo
auroc_save_dir = './'

logger_val = logging.getLogger()
logger_val.setLevel(logging.INFO)
rq_val = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path_val = os.path.dirname("/root/autodl-tmp/chest-EfficientNet/Logs_validation/")
if not os.path.exists(log_path_val):
    os.makedirs(log_path_val)

log_name_val = log_path_val + '/' + rq_val + '.log'  # 设置日志文件名
print(log_name_val)
logfile_val = log_name_val
# 建立一个filehandler来把日志记录在文件里，级别为debug以上
fh_val = logging.FileHandler(logfile_val, mode='w')  # 将日志信息输出到磁盘文件上
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

logger.addHandler(fh_val)

# todo 添加LOSS图像
loss_list = []
epoch_list = []
loss_img = log_name_1

label_dir = './ChestX-ray14/'
image_dir = '/root/autodl-tmp/images'
weight_dir = '/root/autodl-tmp/chest-EfficientNet/weight'

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

trainTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5), #以下三个是随机旋转
    transforms.RandomRotation(degrees=5, resample=False, expand=False, center=None),
    transforms.ColorJitter(contrast=0.1, saturation=0.1, hue=0.1), #gaibianliangdudeng
    # transforms.TenCrop(size=, vertical_flip=False), #裁剪后翻转   若为sequence,则为(h,w)，若为int，则(size,size)  flase，即默认为水平翻转
    # transforms.RandomOrder, #操作顺序随机打乱
    # transforms.LinearTransformation(transformation_matrix)  #baihuachuli
    # transforms.RandomRotation(5), #重复 不用
    transforms.ToTensor(),
    # XTranslation(5,'cyclic'),
    normTransform,
])

valTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normTransform
])


# todo initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)


def weights_init_uniform(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def standradization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    print(mu, sigma)
    return  (data - mu) / sigma



#to do loss function
# def get_loss_function(output, target, loss_weight,epoch=0):
#     possib_vec = 1 / (1 + (-output).exp())
#
#     # loss = -target * (possib_vec + 1e-10).log() - (1 - target) * (1 - possib_vec + 1e-10).log()
#     # loss = -0.25 * ((1 - possib_vec - 1e-10) ** 2) * target * (possib_vec + 1e-10).log() - 0.75 * ((possib_vec + 1e-10) ** 2) * (1 - target) * (1 - possib_vec + 1e-10).log()
#     loss = -((1 - possib_vec - 1e-10) ** 2) * target * (possib_vec + 1e-10).log() - ((possib_vec + 1e-10) ** 2) * (1 - target) * (1 - possib_vec + 1e-10).log()
#     # return 3 * loss.mean()
#     # loss = -0.4*((1 - possib_vec - 1e-10) ** 2) * target * (possib_vec + 1e-10).log() - 0.6*(((possib_vec-0.1) + 1e-10) ** 2) * (
#     #             1 - target) * ((1 -(possib_vec-0.1).clamp(max=1) + 1e-10)).log()
#     # return 3 * loss.mean()
#
#
#     # a = torch.cuda.FloatTensor()
#     # a = a ** 2
#     # loss = loss * a
#     # return 3 * 14 / 3.888 * loss.mean()xiao
#     temp = np.random.randint(1, 500)
#     # if temp == 2:
#     #     print(("原loss" + str(loss.mean())))
#     # # todo 3.31 修改loss
#     loss = torch.mean(torch.sum(loss * loss_weight,dim=1))
#
#     if temp == 2:
#         print(("现loss" + str(loss)))
#     if epoch <15:
#         return 10 * loss
#     elif epoch< 35:
#         return 3 * loss
#     else:
#         return loss
def get_loss_function(output, target, loss_weight,epoch=0,alpha=0.4,gamma_1=1.0,gamma_2=4.0):
    possib_vec = 1 / (1 + (-output).exp())

    FL = -((1 - possib_vec - 1e-10) ** gamma_1) * target * (possib_vec + 1e-10).log() - ((possib_vec + 1e-10) ** gamma_2) * (1 - target) * (1 - possib_vec + 1e-10).log()
    pt = target * possib_vec + (1 - target) * (1 - possib_vec)


    # if alpha >= 0:
    #     alpha_t = alpha * target + (1 - alpha) * (1 - target)
    #     FL = alpha_t * FL

    loss = FL
    temp = np.random.randint(1, 500)
    loss = torch.mean(torch.sum(loss * loss_weight, dim=1))
    if temp == 2:
        print(("现loss" + str(loss)))
    if epoch < 15:
        return 10 * loss
    elif epoch < 35:
        return 3 * loss
    else:
        return loss


# -------------Training------------------#
def train_model(model, optimizer, num_epochs=50):


    loss_weight=1/14*(torch.ones(14).cuda())
    print("loss_weight" +str(loss_weight))

    # my_loss = AsymmetricLossMultiLabel()
    since = time.time()
    # todo
    weight_path = "/root/autodl-tmp/chest-EfficientNet/weight/" + rq + "/"
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # todo jia resume
    # resume = False
    resume = "/root/autodl-tmp/chest-EfficientNet/weight" + "//" + "densenet_epoch_28.pkl"


    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))

            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            epoch_number = int(resume.split("_")[2].split(".")[0])
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            epoch_number = 0
    else:
        print('-------------- New training session ----------------')

    for epoch in range(epoch_number, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(trainLoader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            # todo modify data, target
            # data = data.float()
            # target = target.float()
            optimizer.zero_grad()
            output = model(data)
            # print(output[1:20])
            # print(target[1:20])
            # todo modify loss function
            # loss_girl = focal_loss(alpha=[1], gamma=2, num_classes = 14, size_average=True)
            # pdb.set_trace()
            # loss = loss_girl(output, target)

            # todo
            #
            if epoch == 0:
                loss = get_loss_function(output, target,loss_weight=loss_weight, epoch=epoch)
            else:
                loss = get_loss_function(output, target,loss_weight=loss_weight, epoch=epoch)
            # loss = my_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 200 == 0:

                print(f'Batch Index:{batch_idx}, Batch Loss:{loss}')
                logger.info(f'Batch Index:{batch_idx}, Batch Loss:{loss}')
            running_loss += loss.item()
        epoch_loss = running_loss
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        print( time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
        # todo
        scheduler.step()
        # todo change the place
        # scheduler.step(sum_auroc)

        print('{} Loss: {:.4f}'.format('train', epoch_loss))
        logger.info('epoch{} {} Loss: {:.4f}'.format(epoch, 'train', epoch_loss))
        print("train_loss_weight" + str(loss_weight))
        # todo draw loss image
        loss_list.append(epoch_loss)
        epoch_list.append(epoch)

        plt.title('Train Loss vs. epoches')
        plt.xlabel('epoch')
        plt.ylabel('Train Loss')
        plt.plot(epoch_list, loss_list, 'o-')
        plt.subplot(2, 2, 2)
        plt.savefig(loss_img)

        file_name = os.path.join("/root/autodl-tmp/chest-EfficientNet/weight", rq,
                                 'densenet_epoch_{}.pkl'.format(epoch))

        torch.save({'state_dict': model.state_dict()}, file_name)

        # todo validation
        import torchvision.models as models
        def prediction(val_loader, net):
            # net.eval()
            predict_container = np.zeros((0, 14))
            target_container = np.zeros((0, 14))
            for i, (data, target) in enumerate(val_loader):
                data = Variable(data.float().cuda())
                target = Variable(target.float().cuda())
                output = net(data)
                pred_temp = 1 / (1 + (-output).exp())
                # preds = (pred_temp > 0.5)
                # print preds.data.cpu().numpy()
                predict_container = np.concatenate((predict_container, pred_temp.data.cpu().numpy()), axis=0)
                target_container = np.concatenate((target_container, target.data.cpu().numpy()), axis=0)

            return predict_container, target_container

        def makeValLoader():
            valTransform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                normTransform
            ])
            # todo
            data_val = MultiLabelDataset(label_dir + 'val.csv', image_dir, valTransform)
            # data_val = MultiLabelDataset('./csv/Data_Entry_2017_val_guan.csv', image_dir, valTransform)

            valLoader = DataLoader(
                data_val, batch_size=32, shuffle=False, num_workers=20)
            dataset_val_len = len(data_val)
            print('Validation date set length is ', dataset_val_len)
            return valLoader

        # todo
        # train = pd.read_csv("./weight")
        # np.isnan(train).any()
        # train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

        # def main():
        valLoader = makeValLoader()
        # todo

        # densenet = res2net50_26w_4s_ex(pretrained=True)
        #densenet = SwinTransformer_acmix_t(pretrained=False)
        densenet = SwinTransformer_CAWS_t(pretrained=False)
        #densenet = swin_tiny_patch4_window7_224(pretrained=False)
        # densenet.output = nn.Linear(1024, 14)
        # fc_features = densenet.fc.in_features
        # densenet.fc = nn.Linear(2048, 14)
        # densenet.fc = nn.Conv2d(1024, 14, kernel_size=1,
        #                         stride=1, padding=0, bias=True)
        densenet = densenet.cuda()
        # densenet = DataParallel(densenet)
        auroc_dict = {}
        # todo
        if epoch % 1 == 0 or epoch <=8:
            # for epoch_i in range(50):
            # densenet = pickle.load(open(weight_dir + "/201912191527"+'/densenet_epoch_' + str(epoch_i) + '.pkl', 'rb'))
            # densenet.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
            # todo xiu gai lu jing
            checkpoint = torch.load(file_name)
            # $checkpoint = torch.load(weight_dir + "/201912271629" + '/densenet_epoch_' + str(epoch) + '.pkl')
            densenet.load_state_dict(checkpoint['state_dict'])
            # densenet.eval()
            y_score, y_test = prediction(valLoader, densenet)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(14):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                if epoch <= 5:
                    loss_weight[i]=1/math.pow(roc_auc[i],3)
                elif epoch <= 10:
                    loss_weight[i]=1/math.pow(roc_auc[i],2)
                else:
                    loss_weight[i]=1/math.pow(roc_auc[i],1)

            auroc_dict['epoch_' + str(epoch)] = roc_auc
            loss_weight = F.softmax(loss_weight,dim=0)

            # todo

            sum = 0
            for i in list(roc_auc.values()):
                sum += i
            sum_auroc = sum

            print(epoch, sum_auroc / 14, auroc_dict['epoch_' + str(epoch)])
            print("roc_auc.values()"+str(roc_auc.values())+"\n"+"loss_weight"+str(loss_weight))
            # todo
            logger.info(f"epoch:{epoch}, average_auc:{sum_auroc / 14}", auroc_dict['epoch_' + str(epoch)])

            #scheduler.step(sum_auroc)

            with open(auroc_save_dir + 'auroc_dict.pkl', 'wb') as f:
                pickle.dump(auroc_dict, f)
            # todo zi shi ying loss
            #pdb.set_trace()
            #print(roc_auc.values())
            vali_auc =np.array(list(roc_auc.values()))
            vali_auc = 1.0 / vali_auc
            vali_auc_std = standradization(vali_auc)
            vali_sigmoid = 1.0/(1.0 + np.exp(-vali_auc_std))
            vali_sigmoid = vali_sigmoid**2
            vali_sigmoid_sum = vali_sigmoid.sum()
            def get_loss_function_1(output, target):
                possib_vec = 1 / (1 + (-output).exp())
                # loss = -target * (possib_vec + 1e-10).log() - (1 - target) * (1 - possib_vec + 1e-10).log()
                loss = -((1 - possib_vec - 1e-10) ** 2) * target * (possib_vec + 1e-10).log() - (
                            (possib_vec + 1e-10) ** 2) * (1 - target) * (1 - possib_vec + 1e-10).log()
                # loss = -0.25 * ((1 - possib_vec - 1e-10) ** 2) * target * (possib_vec + 1e-10).log() - 0.75 * ((possib_vec + 1e-10) ** 2) * (1 - target) * (1 - possib_vec + 1e-10).log()
                loss = loss * torch.cuda.FloatTensor(vali_sigmoid)
                #print(vali_sigmoid)
                loss = loss / vali_sigmoid_sum * 14
                # a = torch.cuda.FloatTensor()
                # a = a ** 2
                # loss = loss * a
                # return 3 * 14 / 3.888 * loss.mean()
                return loss.mean()



if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] ="3,0,1,2"
    # from torchsummaryX import summary

    data_train = MultiLabelDataset(label_dir + 'train.csv', image_dir, trainTransform)
    # data_train = MultiLabelDataset('./csv/Data_Entry_2017_train_guan.csv', image_dir, trainTransform)

    trainLoader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)

    densenet = SwinTransformer_CAWS_t(pretrained=True)
    #densenet = SwinTransformer_acmix_t(pretrained=True)
    #densenet = swin_tiny_patch4_window7_224(pretrained=True)
    # fc_features = densenet.fc.in_features
    #densenet.output = nn.Linear(1024, 14)
    # densenet.fc = nn.Conv2d(1024, 14, kernel_size=1,
    #                     stride=1, padding=0, bias=True)
    # densenet = fcanet50(num_classes=1_000, pretrained=False)
    # densenet.classifier = nn.Linear(1024, 14)


    # todo 04.01
    # densenet = densenet.cuda()
    # densenet = DataParallel(densenet)
    # densenet = BalancedDataParallel(16, densenet, dim=0).cuda()
    # densenet = torch.nn.DataParallel(densenet).cuda()
    densenet = densenet.cuda()
    # todo apply initialization
    #densenet.apply(weights_init_normal)
    #densenet.apply(weights_init_uniform)
    #densenet.apply(weights_init)
    parameter = 0
    for param in densenet.parameters():
        parameter += param.data.nelement()
    print('Total trainable parameters are {}'.format(parameter))
    print('Total trainable parameters are {}'.format(parameter))
    optimizer = optim.Adam(densenet.parameters(), lr=1*1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
    # optimizer = optim.Adam(densenet.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5)

    train_model(densenet, optimizer, num_epochs=20)


    # summary(densenet, torch.zeros((1, 3, 224, 224)))
