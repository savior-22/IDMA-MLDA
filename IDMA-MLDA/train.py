#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import numpy as np

from loader import *
from IDMAMLDA import trainHGNNP
from sklearn.model_selection import StratifiedKFold
import statistics
import torch.nn as nn
from Loss import *
import torch
from torch import optim
from models import *
from decoder import *
from evaluation import *
from data_preprocessing import *


def parse_args():
	parser = argparse.ArgumentParser(description="Run IDMA-MLDA")
	parser.add_argument('--data', type=str, default='DMI_random_feat')
	parser.add_argument('--attr', type=bool, default=True)
	parser.add_argument('--dim_f', type=int, default=32)
	parser.add_argument('--dim', type=list, default=[32, 32])
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--epoch', type=int, default=2)
	parser.add_argument('--optimizer', type=str, default='SGD')
	parser.add_argument('--weight_decay', type=float, default=0.0005)
	parser.add_argument('--drop_rate', type=float, default=0.0)
	parser.add_argument('--num_classes', type=int, default=3)
	parser.add_argument('--z_dim', type=int, default=256)
	parser.add_argument('--h_dim', type=int, default=512)
	parser.add_argument('--generative', action='store_true',
						help='whether to use generative classifier')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='momentum')
	parser.add_argument('--weight-cls', type=float, default=10,
						help='loss weight for classification term')
	parser.add_argument('--weight-reconst', type=float, default=1,
						help='loss weight for reconstruction term')
	parser.add_argument('--weight-gmm', type=float, default=0.1,
						help='loss weight for GMM term')
	parser.add_argument('--gamma', type=float, default=0.5,
						help='parameter for scheduler')
	parser.add_argument('--clip-grad', type=float, default=None,
						help='clip gradient norm')
	parser.add_argument('--step-size', default=30, type=int)
	parser.add_argument('--lr-step', type=int, default=40)
	parser.add_argument('--rep-method', type=str, default='get_rep_nums1')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()
	print(args.dim, args.lr, args.lr_step, args.epoch, args.optimizer, args.drop_rate)
	data, feats_dict = load_data(args.data)
	X = data[:, :2]
	y = data[:, -1]
	with open(r'drug_data1.pkl', 'rb') as f:
		drug_smiles_all = pickle.load(f)

	skf = StratifiedKFold(n_splits=10, random_state=3, shuffle=True)
	fold = 0
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		y_train = y_train.reshape(-1, 1)
		y_test = y_test.reshape(-1, 1)
		fold = fold + 1

		print('*********************************************')
		print('This is the ' + str(fold) + 'th times training')
		print('*********************************************')
		train_data = np.hstack((X_train, y_train))
		test_data = np.hstack((X_test, y_test))
		train, test, feats, train_edge_init, test_edge_init, microbe_net = split_train_test(train_data, test_data, feats_dict, True)
		train_class_count = get_class_count1(train)
		test_class_count = get_class_count1(test)

		id_list_d = []
		drug_smiles = {}
		id_pick_d = list(np.unique(train[:, 0]))
		for id in id_pick_d:
			id_list_d.append(drug_smiles_all[id])
		for id1 in range(len(id_list_d)):
			drug_smiles[id1] = id_list_d[id1]
		mic_matrix = np.triu(microbe_net)
		rows, cols = np.indices(mic_matrix.shape)

		new_data = np.concatenate((train, test), axis=0)
		feature_hyg = torch.cat((train_edge_init, test_edge_init), dim=0).to(device)

		nodes = dict()
		n_u, n_i = np.unique(new_data[:, 0]), np.unique(new_data[:, 1])
		nums_u, nums_i = len(n_u), len(n_i)
		nodes['u'], nodes['i'], nodes['n_u'], nodes['n_i'] = n_u, n_i, nums_u, nums_i
		nodes['train'], nodes['test'] = len(train), len(test)

		hygraph = construct_hypergraph(new_data)
		edge_list = construct_edge_list(hygraph, nodes)

		G = Hypergraph(len(hygraph), edge_list)
		lbl_train = torch.tensor(train[:, 2])
		lbl_test = torch.tensor(test[:, 2])

		batch_size = train_edge_init.shape[0]
		num_label = len(np.unique(train[:, 2]))
		decoder = Decoder(batch_size, num_label, h_dim=args.h_dim, z_dim=args.z_dim)
		model = HGNNP(
						decoder, x_dim=len(rows), in_channels=512, hid_channels=256, out_channels=512,
						num_label=num_label, z_dim=args.z_dim, h_dim=args.h_dim, use_bn=True, drop_rate=args.drop_rate
					)

		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		criterion = {'reconst': nn.MSELoss(), 'cls': nn.CrossEntropyLoss(), 'GMM': GMMLoss(num_label, args.z_dim)}
		G = G.to(device)
		lbl_train = lbl_train.to(device)
		lbl_test = lbl_test.to(device)

		model = model.to(device)
		train_logits, train_target, test_logits, test_target, g_macro, f1_macro, acs_a, acc1, acc2, acc3, r_mi, p_ma = trainHGNNP(
			args, model, feature_hyg, microbe_net, lbl_train, lbl_test, optimizer,
			G, criterion, nodes, drug_smiles, new_data, train_class_count, test_class_count, fold)

		train_auc_mic, train_aupr_mic, train_auc_mac, train_aupr_mac, train_F1_mic, train_F1_mac = link_prediction(train_logits, train_target)

		test_auc_mic, test_aupr_mic, test_auc_mac, test_aupr_mac, test_F1_mic, test_F1_mac = link_prediction(test_logits, test_target)
		print('test_auc_macro, test_aupr_macro', test_auc_mac, test_aupr_mac)
		print('test_F1_score_macro', test_F1_mac)