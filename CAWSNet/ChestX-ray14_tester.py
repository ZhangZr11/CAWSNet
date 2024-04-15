# from qiangxianyan_zhao import qiangxianyan
import logging
import numpy as np
import os
from pylab import mpl
import _pickle as pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, precision_recall_curve,average_precision_score
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import cycle
from data_parallel_my_v2 import BalancedDataParallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn,optim
from torch.nn import DataParallel
import math
import shutil
import sys
import random
import logging
import time
from tools import MultiLabelDataset
from numpy import interp
from CAWSNet import SwinTransformer_CAWS_t


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei','KaiTi','SimHei','FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False
my_font = font_manager.FontProperties(fname="/root/autodl-tmp/chest-EfficientNet/原版宋体.ttf")


np.set_printoptions(threshold=None)
auroc_save_dir = '/root/autodl-tmp/chest-EfficientNet/ROC_curve'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = os.path.dirname("/root/autodl-tmp/chest-EfficientNet/Logs_test/")
if not os.path.exists(log_path):
    os.makedirs(log_path)

log_name = log_path + '/'+rq + '.log' #设置日志文件名
print(log_name)
logfile = log_name
# 建立一个filehandler来把日志记录在文件里，级别为debug以上
fh = logging.FileHandler(logfile, mode='w') #将日志信息输出到磁盘文件上
fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

logger.addHandler(fh)

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

label_dir = './ChestX-ray14'
image_dir = '/root/autodl-tmp/images'
weight_dir = '/root/autodl-tmp/chest-EfficientNet/weight'


def prediction(val_loader,net):
    #net.eval()
    predict_container =  np.zeros((0, 14))
    target_container = np.zeros((0, 14))
    for i, (data, target) in enumerate(val_loader):
        data = Variable(data.float().cuda())
        target = Variable(target.float().cuda())
        output = net(data)
        # output = net(data.flip(dims=(2,)))
        output += net(data.flip(dims=(3,)))
        # output = net(data.flip(dims=(2, 3)))
        output = output / 2.0
        pred_temp = 1/(1+(-output).exp())
        #preds = (pred_temp > 0.5)
        #print preds.data.cpu().numpy()
        pred_temp = pred_temp.data.cpu().numpy()
        # for i in range(len(pred_temp)):
        #     print(pred_temp[i])
        #     pred_temp[i] = qiangxianyan(pred_temp[i])
        #     print(pred_temp[i])

        predict_container = np.concatenate((predict_container,pred_temp),axis=0)
        target_container = np.concatenate((target_container,target.data.cpu().numpy()),axis=0)

    return predict_container, target_container



def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]


def makeTestLoader():

	testTransform = transforms.Compose([
	    transforms.Resize(224),
	    transforms.ToTensor(),
	    normTransform
	])

	data_test = MultiLabelDataset(label_dir+'test.csv',image_dir,testTransform)
	# data_test = MultiLabelDataset('./csv/Data_Entry_2017_test_guan.csv', image_dir, testTransform)

	class_names = list(data_test.__one_hot_class__())
	NotFindingIndex = class_names.index('No Finding')
	class_names.pop(NotFindingIndex)
	print(f'Testing one hot convert into class:{class_names}')

	testLoader = DataLoader(
	    data_test, batch_size=256, shuffle=False,num_workers=10)
	dataset_test_len=len(data_test)
	print('Test date set length is ',dataset_test_len)
	return testLoader

def main():
	testLoader = makeTestLoader()

	testTransform = transforms.Compose([
		transforms.RandomCrop(224),
		transforms.ToTensor(),
		normTransform
	])
	data_test = MultiLabelDataset(label_dir + 'test.csv', image_dir, testTransform)
	# data_test = MultiLabelDataset('./csv/Data_Entry_2017_test_guan.csv', image_dir, testTransform)

	class_names = list(data_test.__one_hot_class__())
	NotFindingIndex = class_names.index('No Finding')
	class_names.pop(NotFindingIndex)

	#todo
	#densenet = dense_git.DenseNet()
	# net = densenet_11.DenseNet()
	# densenet = densenet_11_ACB.DenseNet()
	# densenet = densenet_11_DSE.DenseNet()
	# densenet = models.densenet121(pretrained=True)
	# densenet = densenet_121.densenet121(pretrained=False)
	#densenet = SwinTransformer_acmix_t()
	densenet = SwinTransformer_CAWS_t()
	# densenet = shufflenetv2b_w1(pretrained=False)
	# densenet.fc = nn.Linear(42, 14)
	# fc_features = densenet.fc.in_features
	#densenet.output = nn.Linear(1024, 14)
	densenet = densenet.cuda()
	# densenet = BalancedDataParallel(256, densenet, dim=0).cuda()
	# densenet = DataParallel(densenet)

	auroc_dict ={}
	#todo
	#densenet = pickle.load(open(weight_dir + "/201912191527" + '/densenet_epoch_' + str(BEST_EPOCH) + '.pkl', 'rb'))
	#densenet.eval()




	#todo
	checkpoint = torch.load(weight_dir + "/202404131958" + '/densenet_epoch_' + str(BEST_EPOCH) + '.pkl')
	densenet.load_state_dict(checkpoint['state_dict'], False)
	densenet.eval()
	with torch.no_grad():
		y_score, y_test = prediction(testLoader,densenet) # 22389,14

		# binary_predictions = (y_score > 0.5).astype(int)

		# # 计算准确率
		# # accuracy = accuracy_score(target_container, binary_predictions)
		# acc = Accuracy(y_test,binary_predictions)
        # # print(acc)
		#
		# # 计算特异度
		# # 特异度指所有真实负例中，被正确判定为负例的比例
		# tn = ((y_test == 0) & (binary_predictions == 0)).sum()  # 真负例数
		# fp = ((y_test == 0) & (binary_predictions == 1)).sum()  # 假正例数
		# tp = ((y_test == 1) & (binary_predictions == 1)).sum()  # 真正例数
		# fn = ((y_test == 1) & (binary_predictions == 0)).sum()  # 假负例数
		# specificity = tn / (tn + fp)  # 特异度
		# accuracy = tp / (tp + fp)  # 准确率
		# recall = tp / (tp + fn)  # 召回率
		# f1 = 2 * (accuracy * recall) / (accuracy + recall)
		# print(acc)
		# print("Accuracy:", accuracy)
		# print("Specificity:", specificity)
		# print("Recall:", recall)
		# print("F1 Score:", f1)



		# Compute ROC curve and ROC area for each class
		fpr = dict()
		tpr = dict()
		precision = dict()
		recall = dict()
		test_roc_auc = dict()

		for i in range(14):
			fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
			test_roc_auc[i] = auc(fpr[i], tpr[i])
			#ave_presision[i] = average_precision_score(y_test[:, i], y_score[:, i])
			#precision[i], recall[i],_ = precision_recall_curve(y_test[:, i], y_score[:, i])


		#print(len(precision))
		#todo
		sum = 0
		sum_precision = 0

		for i in list(test_roc_auc.values()):
			sum += i
		print("epoch_{}".format(BEST_EPOCH))
		print(test_roc_auc)
		print(sum/14)
		logger.info(f"epoch:{BEST_EPOCH}, test_roc_auc:{test_roc_auc}, average_auc:{sum / 14}")
		#todo
		# with open(auroc_save_dir+'test_auroc_dict.pkl','wb') as f:
		# #with open(auroc_save_dir + '/auroc_dict.pkl', 'wb') as f:
		# 	pickle.dump(test_roc_auc,f)

		ROC_file_name = os.path.join("/root/autodl-tmp/chest-EfficientNet/ROC_curve", rq)

		# 计算微观平均ROC曲线（micro-average ROC curve）和相应AUC值
		# fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
		# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

		# 计算宏观平均ROC曲线（micro-average ROC curve）和相应AUC值
		# First aggregate all false positive rates
		all_fpr = np.unique(np.concatenate([fpr[i] for i in range(14)]))
		# Then interpolate all ROC curves at this points
		mean_tpr = np.zeros_like(all_fpr)
		for i in range(14):
			mean_tpr += interp(all_fpr, fpr[i], tpr[i])
		# Finally average it and compute AUC
		mean_tpr /= 14
		fpr["macro"] = all_fpr
		# output_1 =list(fpr["macro"])
		tpr["macro"] = mean_tpr
		# output_2 = list(tpr["macro"])
		test_roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

		# todo 保存fpr,tpr
		with open('/root/autodl-tmp/chest-EfficientNet/fprtpr/test_all.csv', 'w') as f:
			f.write(str(list(fpr["macro"]))+'\n')
			f.write(str(list(tpr["macro"])))

		lw = 2
		plt.figure()
		# todo
		# plt.figure(figsize=(3.3, 3.3))
		plt.figure(figsize=(3.3, 3.3), dpi=500)
		# 画出微观曲线
		# plt.plot(fpr["micro"], tpr["micro"],
		# 		 label='micro-average ROC curve (area = {0:0.2f})'
		# 			   ''.format(roc_auc["micro"]),
		# 		 color='deeppink', linestyle=':', linewidth=2)

		# 画出宏观曲线
		plt.plot(fpr["macro"], tpr["macro"],
				 label='average curve (AUC:{0:0.3f})'
					   ''.format(test_roc_auc["macro"]),
				 color='navy', linestyle=':', linewidth=2)

		# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
		#todo
		# colors = cycle(['darkred', 'darkorange', 'deeppink', 'darkviolet', 'forestgreen', 'lightseagreen',
		# 				'blue', 'lightsalmon', 'gold', 'grey', 'saddlebrown', 'violet', 'red', 'crimson'])
		# markers = cycle([ 'o', '^', 'v', '<', '>', 's', '+', 'x', 'D', 'd', '1', '2', 'h', 'p', ])
		# for i, color, marker in zip(range(14), colors, markers):
		# 	plt.plot(fpr[i], tpr[i], color=color, marker=marker, lw=1,
		# 			 label='{0} (AUC:{1:0.2f})'.format(class_names[i], test_roc_auc[i]))
		for i in range(14):
			plt.plot(fpr[i], tpr[i], lw=1,
					 label='{0} (AUC:{1:0.3f})'.format(class_names[i], test_roc_auc[i]))

		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		# plt.rcParams['font.sans-serif']=['SimSun']
		# plt.rcParams['font.sans-serif'] = ['SimHei']
		plt.xlabel(u'假阳率', fontsize=8,fontproperties=my_font)
		plt.ylabel(u'真阳率', fontsize=8,fontproperties=my_font)
		plt.title("ChestX-ray14数据集中14种疾病的ROC曲线和AUC值", fontsize=8,fontproperties=my_font)
		font1 = {'family':'Times New Roman', 'weight':'normal', 'size':'5'}
		plt.legend(loc="lower right",prop=font1)
		plt.tick_params(labelsize=6)

		# plt.xlim([0.00, 1.0])
		# plt.ylim([0.00, 1.05])
		# plt.rcParams['font.sans-serif'] = ['SimSun']
		# plt.rcParams['font.sans-serif'] = ['SimHei']
		# plt.rcParams['axes.unicode_minus'] = False
		# plt.xlabel(u'假正例率', fontsize=8)
		# plt.ylabel(u'真正例率', fontsize=8)
		# plt.title("ChestX-ray14数据集上14种疾病的ROC曲线和AUC值", fontsize=8)
		# font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': '5'}
		# plt.legend(loc="lower right", prop=font1)
		# plt.tick_params(labelsize=6)

		plt.savefig(ROC_file_name)

		plt.savefig(ROC_file_name)




#
# def evaluate(val_loader, net):
#     predict_container, target_container = prediction(val_loader, net)
#
#     # 将预测概率转换为二进制预测
#     binary_predictions = (predict_container > 0.5).astype(int)
#
#     # 计算准确率
#     # accuracy = accuracy_score(target_container, binary_predictions)
#
#     # 计算特异度
#     # 特异度指所有真实负例中，被正确判定为负例的比例
#     tn = ((target_container == 0) & (binary_predictions == 0)).sum()  # 真负例数
#     fp = ((target_container == 0) & (binary_predictions == 1)).sum()  # 假正例数
#     tp = ((target_container == 1) & (binary_predictions == 1)).sum()  # 真正例数
#     fn = ((target_container == 1) & (binary_predictions == 0)).sum()  # 假负例数
#     specificity = tn / (tn + fp)  # 特异度
#     accuracy = tp / (tp + fp)  # 准确率
#     recall = tp / (tp + fn)  # 召回率
#     f1 = 2 * (accuracy * recall) / (accuracy + recall)  # F1分数
#     # 计算召回率
#     # recall = recall_score(target_container, binary_predictions,average='micro')
#
#     # 计算F1分数
#     # f1 = f1_score(target_container, binary_predictions,average='micro')
#
#     return accuracy, specificity, recall, f1

# def acc_et_al(densenet, testLoader):
#     with torch.no_grad():
#         accuracy, specificity, recall, f1 = evaluate(testLoader, densenet)
#         print("Accuracy:", accuracy)
#         print("Specificity:", specificity)
#         print("Recall:", recall)
#         print("F1 Score:", f1)


if __name__ =='__main__':
	#todo
	import os
	# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
	for i in range(0, 20):
		BEST_EPOCH = i
		main()

	# BEST_EPOCH = 6
	# main()


























