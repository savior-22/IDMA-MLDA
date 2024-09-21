import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from hypergraph import *
import sys
import numpy
import tqdm
from torch_geometric.nn import GCNConv, GATConv
from decoder import *
numpy.set_printoptions(threshold=sys.maxsize)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class HGNNPConv(nn.Module):

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		bias: bool = True,
		use_bn: bool = False,
		drop_rate: float = 0.0,
		is_last: bool = False
	):
		super().__init__()
		self.is_last = is_last
		self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
		self.act = nn.ReLU(inplace=True)
		self.drop = nn.Dropout(drop_rate)
		self.theta = nn.Linear(in_channels, out_channels, bias=bias)

	def forward(self, X: torch.Tensor, hg: Hypergraph, drop_rate) -> torch.Tensor:
		X = self.theta(X).relu().detach()
		if self.bn is not None:
			X = self.bn(X)
		X = hg.v2v(X, aggr="mean", drop_rate=drop_rate)
		if not self.is_last:
			X = self.drop(self.act(X))
		return X


class HGNNP(nn.Module):
	def __init__(
		self,
		decoder,
		x_dim,
		in_channels,
		hid_channels,
		out_channels,
		num_label,
		z_dim,
		h_dim,
		use_bn: bool = False,
		drop_rate: float = 0.0,
		generative=True
	):
		super(HGNNP, self).__init__()
		self.softmax = nn.Softmax(dim=1)
		self.dropout = nn.Dropout(0.2)
		self.linear3 = nn.Linear(z_dim, 64)
		self.linear6 = nn.Linear(x_dim, 256)
		self.linear9 = nn.Linear(512, 512)
		self.linear14 = nn.Linear(512, 512)
		self.conv21 = GCNConv(46, 256)
		self.classifier = nn.Linear(64, 3)
		self.bn2 = nn.BatchNorm1d(256)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn9 = nn.BatchNorm1d(512)
		self.bn14 = nn.BatchNorm1d(512)
		self.relu = nn.ReLU(inplace=True)
		self.decoder = decoder
		self.generative = generative
		if generative:
			h_dim = h_dim + num_label
		self.fc_mu = nn.Linear(h_dim, z_dim)
		self.fc_logvar = nn.Linear(h_dim, z_dim)
		self.layers = nn.ModuleList()
		self.layers.append(
			HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
		)
		self.layers.append(
			HGNNPConv(hid_channels, out_channels, use_bn=use_bn, is_last=True)
		)
		self.layers1 = nn.ModuleList()
		self.layers1.append(
			HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
		)
		self.layers1.append(
			HGNNPConv(hid_channels, out_channels, use_bn=use_bn, is_last=True)
		)

	def forward(self, drop_rate, num_label, G, y_target, y_bin, nodes, iftrain, drug_smiles,
				feature_hyg, microbe_net, new_data) -> torch.Tensor:
		vector = []
		molecular_feats = []
		m_feats = []
		mic_feat = torch.tensor(microbe_net).to(torch.float32)
		mic_feat = self.linear6(mic_feat)
		mic_feat = self.bn2(mic_feat)
		mic_feat = self.relu(mic_feat)

		for i in range(len(drug_smiles)):
			temp = self.conv21(drug_smiles[i][0], drug_smiles[i][1])
			temp = self.relu(temp)
			temp = torch.mean(temp, dim=0).reshape(1, -1)
			vector.append(temp)
		matrix = torch.cat(vector, dim=0).detach().numpy()

		for idx in range(new_data.shape[0]):
			molecular_feat = matrix[new_data[idx][0]]
			molecular_feats.append(molecular_feat)
			m_feat = mic_feat[new_data[idx][1] - len(drug_smiles)]
			m_feats.append(m_feat)
		molecular_feats = np.array(molecular_feats)
		feature_drug = torch.tensor(molecular_feats)
		feature_mic = torch.stack(m_feats)
		feature_mic = feature_mic.to(torch.float32)

		feature_dual = torch.cat((feature_drug, feature_mic), dim=1)
		feature_dual = self.linear9(feature_dual)
		feature_dual = self.bn9(feature_dual)
		feature_dual = self.relu(feature_dual)

		for layer in self.layers:
			feature_hyg = layer(feature_hyg, G, drop_rate)

		e_1 = feature_dual + feature_hyg
		feature_dual1 = self.linear14(e_1)
		feature_dual1 = self.bn14(feature_dual1).relu()
		feature_hyg1 = e_1
		for layer in self.layers1:
			feature_hyg1 = layer(feature_hyg1, G, drop_rate)

		e_2 = feature_dual1 + feature_hyg1
		if iftrain:
			feat_gmm_init = e_2[:nodes['train']]
		else:
			feat_gmm_init = e_2[nodes['train']:]

		mus, logvars, z1 = self.get_z(feat_gmm_init, y_bin, iftrain)

		if iftrain:
			rep_nums, class_cnts = get_rep_nums1(y_target)
			covar, centers = update_stats(mus, y_target, len(rep_nums))
			many_cls, few_cls = split_class(class_cnts)
			augment_dict = {}
			for c in range(num_label):
				if c in many_cls:
					idx = many_cls.index(c)
					aug_cls_list = many_cls[:idx]
				else:
					aug_cls_list = many_cls
				if len(aug_cls_list) > 0:
					aug_source = get_class_data(y_target, aug_cls_list, cat=True)
					augment_dict[c] = aug_source
			z, target = self.augment_batch(mus, logvars, z1, y_target, rep_nums, augment_dict, centers, covar, iftrain)	# 高斯混合
			logits = self.linear3(z)
			logits = self.bn3(logits)
			logits = self.relu(logits)
			logits = self.classifier(logits)
			x_reconst = self.decoder(z1, y_bin)

		else:
			logits = self.linear3(z1)
			logits = self.bn3(logits)
			logits = self.relu(logits)
			logits = self.classifier(logits)
			x_reconst = self.decoder(z1, y_bin)
			target = y_target

		return mus, logvars, z1, logits, x_reconst, target, feat_gmm_init

	def get_z(self, x, y, iftrain):
		if self.generative:
			x = torch.cat([x, y], dim=-1)
		mu = self.fc_mu(x)
		logvar = self.fc_logvar(x)
		z = self.reparameterize(mu, logvar, iftrain)
		return mu, logvar, z

	def reparameterize(self, mu, logvar,iftrain):
		if iftrain:
			std = (logvar.clamp(-50, 50).exp() + 1e-8) ** 0.5
			torch.manual_seed(0)
			eps = torch.randn_like(logvar)
			return eps * std + mu
		else:
			return mu

	def augment_batch(self, mus, logvars, z1, target: torch.Tensor, rep_nums, augment_dict, centers, covar, iftrain):
		z_final, y_final = [], []

		for i in range(len(rep_nums)):
			idx = target == i
			num_i = idx.count_nonzero()
			z_final.append(z1[idx])

			aug_num = int(num_i * rep_nums[i])
			if aug_num > 0:
				rand_idx = torch.randint(0, len(augment_dict[i]), [aug_num])
				mus_j, logvars_j, target_j = mus[augment_dict[i][rand_idx]], \
											 logvars[augment_dict[i][rand_idx]], \
											 target[augment_dict[i][rand_idx]]
				z_j = self.reparameterize(mus_j, logvars_j, iftrain)
				z_aug_i = centers[i] + (z_j - centers[target_j]) @ covar
				z_final.append(z_aug_i)
			else:
				z_aug_i = []
			y_final.append(torch.full([len(z1[idx]) + len(z_aug_i)], i))
		return torch.cat(z_final, 0), torch.cat(y_final, 0)


def get_rep_nums1(y_train):
	class_cnts = get_class_count(y_train)
	# max_cnt = max(class_cnts)
	# rep_nums = [(max_cnt - class_cnts[i]) / class_cnts[i] for i in range(len(class_cnts))]
	rep_nums = [0, 5, 5]
	return rep_nums, class_cnts


def get_class_count(y_train):
	counts = {}
	y_train = list(y_train)
	for i in range(3):
		counts[i] = y_train.count(i)
	ret = [counts[j] for j in range(len(counts))]
	return ret


def update_stats(mus, target, num_classes):
	zdim = mus.size(1)
	covar = torch.zeros(zdim, zdim).to(mus.device)
	centers = []
	for c in range(num_classes):
		mus_c = mus[target == c]
		cc = mus_c.mean(0, True)
		centers.append(cc.squeeze())
		mus_cc = mus_c - cc
		var = mus_cc.t() @ mus_cc
		covar += var
	U, S, V = torch.pca_lowrank(covar)
	Q = V[:, :zdim // 2]
	QQ = Q @ Q.t()
	# print('QQ', QQ.shape)
	return QQ, torch.stack(centers, dim=0)


def split_class(class_cnts, many_shot_thr=500):
	many_cls = []
	few_cls = []
	for i, cnt in enumerate(class_cnts):
		if cnt > many_shot_thr:
			many_cls.append((cnt, i))
		else:
			few_cls.append((cnt, i))
	return [x for _, x in sorted(many_cls, reverse=True, key=lambda x: x[0])], [x for _, x in sorted(few_cls, reverse=True, key=lambda x: x[0])]


def get_class_data(target, cls, cat=False):
	indices = []
	for c in cls:
		idx = (target == c).nonzero(as_tuple=False).squeeze()
		indices.append(idx)
	if cat:
		indices = torch.cat(indices, dim=0)
	return indices

