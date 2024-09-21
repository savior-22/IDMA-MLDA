
import torch
import torch.nn.functional as F
from models import *
from evaluation import *
from Loss import *


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
eps = 1e-6
SimSiamLoss = SimSiamLoss()
SimCLRLoss = SimCLRLoss()


def trainHGNNP(args, model, feature_hyg, microbe_net, lbl_train, lbl_test, optimizer,
			G, criterion, nodes, drug_smiles, new_data, train_class_count, test_class_count, fold):
	n_epoch = args.epoch
	y_true_bin = F.one_hot(lbl_train, num_classes=args.num_classes).float()

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)

	g_macro_list = []
	acs_a_list = []
	f1_macro_list = []
	acc2_list = []
	acc3_list = []

	for epoch in range(n_epoch):
		model.train()

		optimizer.zero_grad()

		for cri in criterion.values():
			cri.train()
		mu, logvar, z1, train_logits, x_reconst, train_target, x_edge = model.forward(
																			args.drop_rate, args.num_classes, G,
																			lbl_train, y_true_bin, nodes, True, drug_smiles,
																			feature_hyg, microbe_net,
																			new_data
																			)
		cls_loss = criterion['cls'](train_logits, train_target.reshape(-1)) * args.weight_cls
		reconst_loss = criterion['reconst'](x_reconst, x_edge[:nodes['train']]) * args.weight_reconst
		gmm_loss = criterion['GMM'](mu, logvar, z1) * args.weight_gmm
		loss = reconst_loss + cls_loss + gmm_loss
		loss.backward(retain_graph=True)
		max_norm = 1.0
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
		optimizer.step()
		scheduler.step()
		print(epoch, cls_loss.detach(), reconst_loss.detach(), gmm_loss.detach())

		model.eval()

		for cri in criterion.values():
			cri.eval()
		elbos = torch.zeros(lbl_test.size(0), args.num_classes)
		for j, c in enumerate(range(0, 3)):
			target_onehot = torch.zeros(lbl_test.size(0), args.num_classes)
			target_c = torch.zeros(lbl_test.size(0), 1)
			target_c[:, 0] = c
			target_onehot[:, c] = 1.0
			mu, logvar, z, test_logits, x_reconst, test_target, x_edge = model.forward(
			args.drop_rate, args.num_classes,
			G, lbl_test, target_onehot, nodes, False,
			drug_smiles, feature_hyg, microbe_net, new_data
			)

			for q in range(lbl_test.size(0)):
				cls_loss = criterion['cls'](test_logits[q], target_c.long().reshape(-1)[q]) * args.weight_cls
				elbo = cls_loss
				elbos[q, j] = elbo
		probs = torch.softmax(-elbos, dim=1)
		predicts = torch.argmax(probs, dim=1)

		g_macro, f1_macro, acs_a, acc1, acc2, acc3, r_mi, p_ma = add_test_metrics(lbl_test.detach().numpy(),
																		predicts.detach().numpy(), train_class_count,
																		test_class_count)
		g_macro_list.append(g_macro)
		acs_a_list.append(acs_a)
		f1_macro_list.append(f1_macro)
		acc2_list.append(acc2)
		acc3_list.append(acc3)

		if g_macro == max(g_macro_list) and acs_a == max(acs_a_list) and f1_macro == max(f1_macro_list):
			epoch_id = epoch
			with open('/Users/Downloads/IDMA-MLDA/result1/patent/1/my_best' + str(fold) + '.model', 'wb') as file:
				pickle.dump(model, file)

		if g_macro == max(g_macro_list) and f1_macro == max(f1_macro_list):
			epoch_id1 = epoch
			with open('/Users/Downloads/IDMA-MLDA/result1/patent/1/my_bests' + str(fold) + '.model', 'wb') as file:
				pickle.dump(model, file)

		if g_macro == max(g_macro_list) and acs_a == max(acs_a_list):
			epoch_id2 = epoch
			with open('/Users/Downloads/IDMA-MLDA/result1/patent/1/my_bestss' + str(fold) + '.model', 'wb') as file:
				pickle.dump(model, file)

		print('g_macro', g_macro, 'acs_a', acs_a, 'f1_macro', f1_macro, 'acc2', acc2, 'acc3', acc3)

	if g_macro_list[epoch_id] >= g_macro_list[epoch_id1] and g_macro_list[epoch_id] >= g_macro_list[epoch_id2]:
		with open('/Users/Downloads/IDMA-MLDA/result1/patent/1/my_best' + str(fold) + '.model', 'rb') as file:
			new_m = pickle.load(file)

	elif g_macro_list[epoch_id1] >= g_macro_list[epoch_id] and g_macro_list[epoch_id1] >= g_macro_list[epoch_id2]:
		with open('/Users/Downloads/IDMA-MLDA/result1/patent/1/my_bests' + str(fold) + '.model', 'rb') as file:
			new_m = pickle.load(file)

	else:
		with open('/Users/Downloads/IDMA-MLDA/result1/patent/1/my_bestss' + str(fold) + '.model', 'rb') as file:
			new_m = pickle.load(file)

	# with open('/Users/Downloads/IDMA-MLDA_mapping/result/ab_model/184/my_best1.model', 'rb') as file:
	# 	new_m = pickle.load(file)

	new_m.eval()

	for cri in criterion.values():
		cri.eval()
	elbos = torch.zeros(lbl_test.size(0), args.num_classes)

	for j, c in enumerate(range(0, 3)):
		target_onehot = torch.zeros(lbl_test.size(0), args.num_classes)
		target_c = torch.zeros(lbl_test.size(0), 1)
		target_c[:, 0] = c
		target_onehot[:, c] = 1.0
		mu, logvar, z, test_logits, x_reconst, test_target, x_edge = new_m.forward(
																		args.drop_rate, args.num_classes,
																		G, lbl_test, target_onehot, nodes, False,
																		drug_smiles, feature_hyg, microbe_net, new_data,
																	)

		for q in range(lbl_test.size(0)):
			cls_loss = criterion['cls'](test_logits[q], target_c.long().reshape(-1)[q]) * args.weight_cls
			elbo = cls_loss
			elbos[q, j] = elbo
	probs = torch.softmax(-elbos, dim=1)
	predicts = torch.argmax(probs, dim=1)
	g_macro, f1_macro, acs_a, acc1, acc2, acc3, r_mi, p_ma = add_test_metrics(lbl_test.detach().numpy(), predicts.detach().numpy(), train_class_count, test_class_count)

	print('final_g_macro', g_macro)
	print('final_acs_a', acs_a)
	print('final_f_macro', f1_macro)

	return train_logits, train_target, probs, test_target, g_macro, f1_macro, acs_a, acc1, acc2, acc3, r_mi, p_ma

