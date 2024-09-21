#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

def load_data(data):
	if data == "DMI":
		data = np.loadtxt('data/DMI/DMI.txt', dtype=int)
		feats_u = np.loadtxt('data/DMI/DMI_featd.txt', dtype=float)
		feats_v = np.loadtxt('data/DMI/DMI_featm.txt', dtype=float)
		feat_dict = {'u': feats_u, 'v': feats_v}
		return data, feat_dict
	elif data == 'DMI_random_feat':
		data = np.loadtxt('data/DMI_random_feat/DMI.txt', dtype=int)
		feats_u = np.loadtxt('data/DMI_random_feat/DMI_featd.txt', dtype=float)
		feats_v = np.loadtxt('data/DMI_random_feat/DMI_featm.txt', dtype=float)
		feats_v_sim = np.loadtxt('data/DMI_random_feat/DMI_featm_sim.txt', dtype=float)
		feat_dict = {'u': feats_u, 'v': feats_v, 'v_sim': feats_v_sim}
		return data, feat_dict


def split_train_test(train, test, feats_dict, flag):
	train_nodes_u = [i for i in list(set(train[:, 0]))]
	train_nodes_i = [i for i in list(set(train[:, 1]))]
	feats = {'u': [], 'v': [], 'v_sim': []}
	microbe_sim = np.copy(np.array(feats_dict['v_sim']))
	ind = np.arange(feats_dict['v_sim'].shape[0])
	train_ui = train_nodes_u

	for i in train_nodes_i:
		train_ui.append(i)

	u_train, i_train = np.unique(train[:, 0]), np.unique(train[:, 1])
	f_test = []
	for line in test:
		if line[0] in u_train and line[1] in i_train:
			f_test.append(line)
	f_test = np.array(f_test)

	idx_u_map = {j: i for i, j in enumerate(np.unique(train[:, 0]))}
	idx_i_map = {j: len(np.unique(train[:, 0]))+i for i, j in enumerate(np.unique(train[:, 1]))}

	new_train, new_test = [], []

	for line in train:
		tmp = []
		tmp.append(idx_u_map[line[0]])
		tmp.append(idx_i_map[line[1]])
		tmp.append(line[2])
		new_train.append(tmp)
	for line in f_test:
		tmp = []
		tmp.append(idx_u_map[line[0]])
		tmp.append(idx_i_map[line[1]])
		tmp.append(line[2])
		new_test.append(tmp)
	new_train, new_test = np.array(new_train), np.array(new_test)
	print('number of training data,number of testing data', len(np.unique(new_train[:, 0])), len(np.unique(new_train[:, 1])))
	indices_to_keep = []
	if flag == True:
		for i in train_ui:
			if i < feats_dict['u'].shape[0]:
				feats['u'].append(feats_dict['u'][i])
			else:
				feats['v'].append(feats_dict['v'][i - feats_dict['u'].shape[0]])
				indices_to_keep.append(i - feats_dict['u'].shape[0])
		indices_to_remove = np.setdiff1d(ind, indices_to_keep)
		microbe_net = np.delete(microbe_sim, indices_to_remove, axis=0)
		microbe_net = np.delete(microbe_net, indices_to_remove, axis=1)
		feats['u'] = np.array(feats['u'])
		feats['v'] = np.array(feats['v'])

		x_edge_init = torch.zeros(0, feats['u'].shape[1] + feats['v'].shape[1])
		test_edge_init = torch.zeros(0, feats['u'].shape[1] + feats['v'].shape[1])

		for edge in new_train:
			node1, node2 = edge[0], edge[1]
			x_edge_temp = torch.cat((torch.tensor(feats['u'][node1]).reshape(1, -1), torch.tensor(feats['v'][node2 - feats['u'].shape[0]]).reshape(1, -1)), dim=1)
			x_edge_init = torch.cat((x_edge_init, x_edge_temp), dim=0).float()

		for edge in new_test:
			node1, node2 = edge[0], edge[1]
			test_edge_temp = torch.cat((torch.tensor(feats['u'][node1]).reshape(1, -1), torch.tensor(feats['v'][node2 - feats['u'].shape[0]]).reshape(1, -1)), dim=1)
			test_edge_init = torch.cat((test_edge_init, test_edge_temp), dim=0).float()

		return new_train, new_test, feats, x_edge_init, test_edge_init, microbe_net

	elif flag == False:
		return new_train, new_test, None


def construct_hypergraph(data):
	t_data = {}
	for i, line in enumerate(data):
		t_data[i] = [line[0], line[1]]
	return t_data


def construct_edge_list(data, nodes):
	num_edge = nodes['n_u'] + nodes['n_i']
	edge_list = [[] for _ in range(num_edge)]    #[[],[],[],[]]
	for i in range(num_edge):
		for key, value in data.items():
			if i in value:
				edge_list[i].append(int(key))		# [[]]
	print('edge num', len(edge_list))
	return edge_list


def get_class_count1(y_train):
	counts = {}
	y_train = list(y_train[:, 2])
	for i in range(3):
		counts[i] = y_train.count(i)
		# print('count', counts[i])
	ret = [counts[j] for j in range(len(counts))]
	return ret