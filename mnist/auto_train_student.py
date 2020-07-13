import subprocess
from glob import glob

# fls = glob("./")

data_size_list = ['100', '1000', '3000', '10000']
data_seed_list = ['1']  # , '2', '3']
dropout_list = ['0.01', '0.1', '0.3', '0.5', '0.7', '0.9', '0.99']
# Penalty
# Batch Norm
student_do = '0.3'
T = '1'
alpha = '0.1'

base_data_path = '/home/temp_use/data/'
base_pred_path = './predict/'

fl = 'distill_mnist.py'
for d_size in data_size_list:
    for do in dropout_list:
        for d_seed in data_seed_list:
            data_name = d_size + '_s' + d_seed
            checkpoint_name = data_name + '-' + do + 'do' + '_bn'
            save_name = checkpoint_name + '-student_' + 'do' + student_do + '_bn'
            subprocess.call(['python', fl, '--epochs', '50', '--batch-size', '128', '--lr', '0.01', '--hidden', '500',
                             '--tensorboard', '--batch-norm', '--gpu', '1',
                             '--seed', d_seed, '--data', 'preprocessed_data/' + data_name + '.pt',
                             '--T', T, '--alpha', alpha, '--dropout', student_do,
                             '--checkpoint', 'model/' + checkpoint_name + '.pt',
                             '--save', 'model/' + save_name])