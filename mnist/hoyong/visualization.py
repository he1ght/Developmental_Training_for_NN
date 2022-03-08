import time
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product
from collections import Counter
from sklearn.manifold import TSNE

MODEL_DIR = Path('.') / 'model'
DATA_DIR = Path('.') / 'data'
RES_DIR = Path('.') / 'result'

TAG_DATA = ['restricted_uniform', 'uniform']
NUM_DATA = ['6T', '10T', '60T']

file_nickname_map = \
		{'10E_FF_500h_0.1do_1e-5L1_1e-6L2_PT_0.1a': 'Tf_self',
		 'BAN_500h_0.1do_1e-5L1_1e-6L2_k0': 'Base',
		 'FF_500h_0.99do_skd_0.1do_1e-5L1_1e-6L2_bn_0.005a': 'mDT-KD',
		 'FF_500h_5e-4L2_bn': 'L2',
		 'FF_500h_5e-5L1_bn': 'L1',
		 'FF_500h_0.01LS_bn': 'LS',
		 'Tf-reg_500h_0.1do_1e-5L1_1e-6L2': 'Tf_reg',
		 'FF_500h_0.1do_1e-5L1_1e-6L2_0.1LS_bn': 'LS^',
		 'FF_500h_0.4do_bn': 'Dropout',
		 'FF_500h_0.99do_skd_0.1do_1e-5L1_1e-6L2_bn_0.2-a-ada': 'mDT-KD_ada',
		 'BAN_500h_0.1do_1e-5L1_1e-6L2_k1': 'BAN',
		 'RI_FF_500h_0.1do_1e-5L1_1e-6L2_skd_0.1a': 'RI-KD',
		 'SimpleKD_500h_0.1do_1e-5L1_1e-6L2': 'Self-KD',
		 'GT': 'GT'}

def main():
	model_names = load_listdir(MODEL_DIR)

	for model_name in model_names:
	
		out_filepath_mnist = RES_DIR / '{}_mnist.pkl'.format(model_name.stem)
		with open(out_filepath_mnist, 'rb') as f:
			out_mnist = pkl.load(f)

		out_filepath_mnist_test = RES_DIR / '{}_mnist_test.pkl'.format(model_name.stem)
		with open(out_filepath_mnist_test, 'rb') as f:
			out_mnist_test = pkl.load(f)

		seed_num = model_name.stem.split('-')[-1]
		if seed_num == '1':
			continue

		for num, tag in product(NUM_DATA, TAG_DATA):
			# seed_num = model_name.stem.split('-')[-1]
			# if seed_num != '1':
			# 	continue

			out_filepath_toy = RES_DIR / '{}_{}_{}.pkl'.format(model_name.stem, tag, num)
			with open(out_filepath_toy, 'rb') as f:
				out_toy = pkl.load(f)

			print(tag, num, model_name, end=' ')
			fig_path = Path('.') / 'fig/{}_{}_{}_seed-{}.jpg'.format(tag, num, file_nickname_map[model_name.stem[:-7]], seed_num)
			visualization(out_mnist, out_toy, fig_path=fig_path)
			# fig_path = Path('.') / 'fig/toy_{}_{}_{}.jpg'.format(tag, num, file_nickname_map[model_name.stem[:-7]])
			# visualization_toy(out_toy, fig_path=fig_path)

		# fig_path = Path('.') / 'fig/test_{}.jpg'.format(file_nickname_map[model_name.stem[:-7]])
		# visualization_test(out_mnist, out_mnist_test, fig_path=fig_path)

	
def load_listdir(dirpath):
	tmp = []
	for filename in sorted(dirpath.iterdir()):
		tmp.append(filename)
	return tmp


def visualization(out_mnist, out_toy, fig_path='figure.jpg'):
	out_mnist = out_mnist.detach().cpu().numpy()
	out_toy = out_toy.detach().cpu().numpy()

	out_total = np.vstack((out_mnist, out_toy))

	lbl_mnist = np.argmax(out_mnist, axis=1)
	lbl_toy = np.argmax(out_toy, axis=1)

	t_start = time.time()
	tsne = TSNE(n_components=2, random_state=1)
	out_total_res = tsne.fit_transform(out_total)
	t_end = time.time()
	print("[ Elapsed time: {:.4f} s ]".format(t_end - t_start))

	out_mnist_res = out_total_res[:out_mnist.shape[0]]
	out_toy_res = out_total_res[out_mnist.shape[0]:]

	plt.figure(figsize=(60, 40))
	for i in range(10):
		plt.plot(out_mnist_res[np.where(lbl_mnist == i)][:, 0],
				 out_mnist_res[np.where(lbl_mnist == i)][:, 1],
				 'x', label=str(i), alpha=0.3)
	for i in range(10):
		plt.plot(out_toy_res[np.where(lbl_toy == i)][:, 0],
				 out_toy_res[np.where(lbl_toy == i)][:, 1],
				 '.', label=str(i), alpha=0.3)
	plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='xx-large')
	plt.savefig(fig_path)
	plt.close()


def visualization_toy(out_toy, fig_path='figure.jpg'):
	out_toy = out_toy.detach().cpu().numpy()

	lbl_toy = np.argmax(out_toy, axis=1)

	t_start = time.time()
	tsne = TSNE(n_components=2, random_state=1)
	out_toy_res = tsne.fit_transform(out_toy)
	t_end = time.time()
	print("[ Elapsed time: {:.4f} s ]".format(t_end - t_start))

	plt.figure(figsize=(60, 40))
	for i in range(10):
		plt.plot(out_toy_res[np.where(lbl_toy == i)][:, 0],
				 out_toy_res[np.where(lbl_toy == i)][:, 1],
				 '.', label=str(i), alpha=0.3)
	plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='xx-large')
	plt.savefig(fig_path)
	plt.close()

def visualization_test(out_mnist, out_mnist_test, fig_path='figure.jpg'):
	out_mnist = out_mnist.detach().cpu().numpy()
	out_mnist_test = out_mnist_test.detach().cpu().numpy()

	out_total = np.vstack((out_mnist, out_mnist_test))

	lbl_mnist = np.argmax(out_mnist, axis=1)
	lbl_mnist_test = np.argmax(out_mnist_test, axis=1)

	t_start = time.time()
	tsne = TSNE(n_components=2, random_state=1)
	out_total_res = tsne.fit_transform(out_total)
	t_end = time.time()
	print("[ Elapsed time: {:.4f} s ]".format(t_end - t_start))

	out_mnist_res = out_total_res[:out_mnist.shape[0]]
	out_mnist_test_res = out_total_res[out_mnist.shape[0]:]

	plt.figure(figsize=(60, 40))
	for i in range(10):
		plt.plot(out_mnist_res[np.where(lbl_mnist == i)][:, 0],
				 out_mnist_res[np.where(lbl_mnist == i)][:, 1],
				 'x', label=str(i), alpha=0.3)
	for i in range(10):
		plt.plot(out_mnist_test_res[np.where(lbl_mnist_test == i)][:, 0],
				 out_mnist_test_res[np.where(lbl_mnist_test == i)][:, 1],
				 '.', label=str(i), alpha=0.3)
	plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='xx-large')
	plt.savefig(fig_path)
	plt.close()

if __name__=="__main__":
	main()

