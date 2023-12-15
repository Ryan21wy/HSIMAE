# References:
# SVM_RBF: https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/global_module/network.py


import os
import itertools
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.image as mi
from Compared_Model_Test import get_data_path, data_trans, get_data_set, label_to_colormap, spilt_dataset


def seed_everything(seed):
    np.random.seed(seed)


class svm_rbf():
    def __init__(self):
        self.name = 'SVM_RBF'
        self.best_est = None

    def parameter_selection(self, trainx, trainy, para_c, para_g, training_ratio=0.5):
        parameters = itertools.product(para_c, para_g)
        train_data, train_gt, val_data, val_gt = spilt_dataset(trainx, trainy, training_ratio=training_ratio)
        best_c = 0
        best_g = 0
        best_metric = 0
        for para in parameters:
            svm = SVC(C=para[0], gamma=para[1], kernel='rbf')
            svm.fit(np.array(train_data), train_gt)
            pred = svm.predict(np.array(val_data))

            oa = metrics.accuracy_score(val_gt, pred)
            aa = np.mean(metrics.recall_score(val_gt, pred, average=None))
            kappa = metrics.cohen_kappa_score(val_gt, pred)
            metric = oa + aa + kappa
            if metric > best_metric:
                best_c = para[0]
                best_g = para[1]
                best_metric = metric
        svm = SVC(C=best_c, gamma=best_g, kernel='rbf')
        svm.fit(np.array(train_data), train_gt)
        return svm, best_c, best_g

    def train(self, trainx, trainy, seed=42):
        cost = []
        gamma = []
        for i in range(-3, 10, 2):
            cost.append(np.power(2.0, i))
        for i in range(-5, 4, 2):
            gamma.append(np.power(2.0, i))

        _, bestc, bestg = self.parameter_selection(trainx, trainy, cost, gamma, 0.5)

        print(bestc, bestg)
        tmpc = [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0,
                0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cost = []
        gamma = []
        for i in tmpc:
            cost.append(bestc * np.power(2.0, i))
            gamma.append(bestg * np.power(2.0, i))
        svm, bestc, bestg = self.parameter_selection(trainx, trainy, cost, gamma, 0.5)
        print(bestc, bestg)
        self.best_est = svm

    def test(self, testx, testy, gt, save_path=None):
        pred = self.best_est.predict(testx)
        pred = pred.reshape(gt.shape)
        colormap_all = label_to_colormap(pred)

        pred[gt == 0] = 0
        colormap = label_to_colormap(pred)

        gt_ = testy.reshape(-1)
        gt_label = gt_[gt_ != 0] - 1

        pred_ = pred.reshape(-1)
        pred_label = pred_[gt_ != 0] - 1

        # cm = metrics.confusion_matrix(gt_label, pred_label)
        oa = metrics.accuracy_score(gt_label, pred_label)
        aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
        kappa = metrics.cohen_kappa_score(gt_label, pred_label)
        ca = metrics.recall_score(gt_label, pred_label, average=None)

        if save_path is not None:
            mi.imsave(os.path.join(save_path, self.name + '_all_oa_' + str(np.around(oa * 100, 2)) + '.png'), colormap_all, cmap='tab20')
            mi.imsave(os.path.join(save_path, self.name + '_oa_' + str(np.around(oa * 100, 2)) + '.png'), colormap, cmap='tab20')
        return oa, aa, kappa, ca


seeds = [3407, 3408, 3409, 3410, 3411]

dataset = 'Houston2013'
data_path, gt_path = get_data_path(dataset)

save_path = r'D:\dataset\HSIMAE\results\compared_results/' + dataset
if not os.path.exists(save_path):
    os.makedirs(save_path)

HSI_data = data_trans(data_path, norm=(1, 0))

label_num = [20, 40, 60, 80]

for l_num in label_num:
    test_results = []
    test_results_per_class = []

    for i in range(5):
        seed_everything(seeds[i])
        train_set, train_gt, test_set, test_gt, all_gt = get_data_set(HSI_data,
                                                                      gt_path,
                                                                      patch_size=1,
                                                                      num=l_num,)
                                                                      # mask=r"D:\dataset\HSIMAE\Dataset\WHU-Hi-LongKou\Train100.npy")

        SVM = svm_rbf()

        train_x = np.array(train_set).squeeze()
        SVM.train(train_x, train_gt, seed=seeds[i])

        testx = np.array(test_set).squeeze()
        oa, aa, kappa, ca = SVM.test(testx, test_gt, all_gt)

        test_results.append([oa, aa, kappa])
        print(test_results)
        test_results_per_class.append(ca)

    test_results = np.array(test_results)
    test_mean = np.mean(test_results, axis=0) * 100
    test_std = np.std(test_results, axis=0) * 100

    class_accuracy_mean = np.mean(test_results_per_class, axis=0) * 100
    class_accuracy_std = np.std(test_results_per_class, axis=0) * 100


    print('parameter is: ')
    print(l_num)

    # print('class_accuracy:')
    # for ca in class_accuracy_mean:
    #     print(np.around(ca, 2))

    print('test oa, aa, kappa:')
    for tm in test_mean:
        print(np.around(tm, 2))
    for ts in test_std:
        print(np.around(ts, 2))
