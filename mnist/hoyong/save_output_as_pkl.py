import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pickle as pkl
from pathlib import Path
from itertools import product

MODEL_DIR = Path('.') / 'model_07'
DATA_DIR = Path('.') / 'data'
RES_DIR = Path('.') / 'result_07'


class ffn_two_layers(nn.Module):
	def __init__(self, hidden=500, dropout=0.1, batch_norm=False):
		super(ffn_two_layers, self).__init__()
		self.hidden = hidden
		self.dropout = dropout
		self.batch_norm = batch_norm

		if self.batch_norm:
			self.bn1 = nn.BatchNorm1d(self.hidden)
			self.bn2 = nn.BatchNorm1d(self.hidden)

		self.fc1 = nn.Linear(28 * 28, self.hidden)
		self.fc2 = nn.Linear(self.hidden, self.hidden)
		self.fc3 = nn.Linear(self.hidden, 10)

	def forward(self, x):
		x = x.view(-1, 28 * 28)
		x = self.fc1(x)
		if self.batch_norm:
			x = self.bn1(x)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.fc2(x)
		if self.batch_norm:
			x = self.bn2(x)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.fc3(x)
		return x

class ToyUniformDataset(torch.utils.data.Dataset):
	def __init__(self, x):
		self.x_data = x
		self.y_data = [0 for _ in range(x.size(0))] # Y는 Dummy label.

	def __len__(self):
		return len(self.x_data)

	def __getitem__(self, idx):
		x = torch.FloatTensor(self.x_data[idx])
		y = torch.FloatTensor(self.y_data[idx])
		return x, y


class Normalize(torch.nn.Module):
	def __init__(self, mean, std):
		super(Normalize, self).__init__()
		self.mean = torch.tensor(mean)
		self.std = torch.tensor(std)

	def forward(self, input):
		x = input / 255.0
		# print(x)
		x = x - self.mean
		x = x / self.std
		return x

def main(model_name, data_dir=None):
	# model load
#	model = torch.load(model_name) # class: ffn_two_layers
	meta = torch.load(model_name, map_location=lambda storage, loc: storage.cpu())
	args = meta['args']
	model = ffn_two_layers(hidden=args['hidden'], dropout=args['dropout'], batch_norm=args['batch_norm'])
	model.load_state_dict(meta['model'])

	# data load
	if not data_dir is None:
		fit_dataset = torch.load(data_dir)
	else:
		fit_dataset = datasets.MNIST('../data_mnist', train=True, download=True, transform=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		]))

	kwargs = {'num_workers': 0, 'pin_memory': True} if args['cuda'] else {}
	data_loader = torch.utils.data.DataLoader(
		fit_dataset,
		batch_size=10000000, shuffle=False, **kwargs)

	for data, target in data_loader:
		output = model(data)
		output = torch.softmax(output, dim=-1)
		output = torch.argmax(output, dim=-1)

	tag_model = model_name.stem
	tag_data = 'mnist' if data_dir is None else data_dir.stem
	out_filepath = RES_DIR /'{}_{}.pkl'.format(tag_model, tag_data)
	with open(out_filepath, 'wb') as f:
		pkl.dump(output, f)


def load_listdir(dirpath):
	tmp = []
	for filename in sorted(dirpath.iterdir()):
		tmp.append(filename)
	return tmp


if __name__=="__main__":
	model_names = load_listdir(MODEL_DIR)
	data_dirs = [None]
	data_dirs.extend(load_listdir(DATA_DIR))

	for model_name, data_dir in product(model_names, data_dirs):
		print(model_name, data_dir)
		main(model_name, data_dir=data_dir)
	
#	main(model_name, data_dir)

