#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch


def link_prediction(scores, target):

	y_true = np.array(target)
	y_scores = scores.detach().numpy()
	y_predict = np.argmax(y_scores, axis=1)

	y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

	CM = confusion_matrix(y_true, y_predict)
	print('confusion matrix is:')
	print(CM)

	CR = classification_report(y_true, y_predict, labels=[0, 1, 2], target_names=['Decrease', 'Increase', 'No sig'])
	print('classfication report is:')
	print(CR)

	auc_micro = roc_auc_score(y_true_bin, y_scores, labels=[0, 1, 2], average='micro')
	aupr_micro = average_precision_score(y_true_bin, y_scores, average='micro')
	auc_macro = roc_auc_score(y_true_bin, y_scores, labels=[0, 1, 2], average='macro')
	aupr_macro = average_precision_score(y_true_bin, y_scores, average='macro')
	F1_score_micro = f1_score(y_true, y_predict, labels=[0, 1, 2], average='micro')
	F1_score_macro = f1_score(y_true, y_predict, labels=[0, 1, 2], average='macro')
	return auc_micro, aupr_micro, auc_macro, aupr_macro, F1_score_micro, F1_score_macro


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def _prf_divide(numerator, denominator):
	mask = denominator == 0.0
	denominator = denominator.copy()
	denominator = np.where(mask, 1, denominator)
	# denominator[mask] = 1  # avoid infs/nans
	result = numerator / denominator
	return result


def add_test_metrics(y_true, y_pred, train_class_count, test_class_count):
	y_true, y_pred = y_true.astype(np.int32), y_pred.astype(np.int32)
	cnf_matrix = confusion_matrix(y_true, y_pred)

	FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
	FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
	TP = np.diag(cnf_matrix)
	TN = cnf_matrix.sum() - (FP + FN + TP)

	recall = _prf_divide(TP, (TP + FN))
	specificity = _prf_divide(TN, (FP + TN))
	precision = _prf_divide(TP, (TP + FP))
	g_marco = ((recall * specificity) ** 0.5).mean()
	cs_accuracy = _prf_divide(TP, (cnf_matrix.sum(axis=1))).mean()
	f1_macro = _prf_divide(2 * precision * recall, precision + recall).mean()

	acc1, acc2, acc3 = shot_acc(TP, train_class_count, test_class_count)

	r_mi = TP[1] / (TP[1] + FN[1])
	p_ma = TP[0] / (TP[0] + FP[0])
	return g_marco, f1_macro, cs_accuracy, acc1, acc2, acc3, r_mi, p_ma


def shot_acc(class_correct, train_class_count, test_class_count, many_shot_thr=500):
	acc1_shot = []
	acc2_shot = []
	acc3_shot = []
	acc1_shot.append((class_correct[0] / test_class_count[0]))
	acc2_shot.append((class_correct[1] / test_class_count[1]))
	acc3_shot.append((class_correct[2] / test_class_count[2]))

	return np.mean(acc1_shot), np.mean(acc2_shot), np.mean(acc3_shot)