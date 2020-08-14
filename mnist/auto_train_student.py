
import subprocess
from glob import glob

max_gpu = 3
dos = ['0', '0.1', '0.3', '0.5']
l2s = ['0', '0', '0', '0']

import os

def child(gpu, do, lrate):
    epochs = '100'
    fl = 'distill_mnist.py'
    data_name = '3000_s1'
    checkpoint_name = 'dummy_teacher'
    alpha = '0.1'
    T = '1'
    save_name = checkpoint_name + '-student_' + 'do' + do  + '_500h' + lrate + 'L2'
    subprocess.call(['python', fl, '--epochs', epochs, '--batch-size', '128', '--lr', '0.1', '--hidden', '500',
                                 '--tensorboard', '--gpu', gpu, '--lrate', lrate, #'--batch-norm',
                                 '--seed', '1', '--data', 'preprocessed_data/' + data_name + '.pt',
                                 '--T', T, '--alpha', alpha, '--dropout', do,
                                 '--checkpoint', 'model/' + checkpoint_name + '.pt',
                                 '--save', 'model/' + save_name])
    os._exit(0)  

child_pids = []
def parent(child_pids):
    for index in range(max_gpu + 1):
        newpid = os.fork()
        if newpid == 0:
            child(str(index), dos[index], l2s[index])
        else:
            child_pids.append(newpid)


parent(child_pids)
for child_pid in child_pids:
    pid, status = os.waitpid(child_pid, 0)

# child_pids = []
# parent(child_pids, '0.01')
# for child_pid in child_pids:
#     pid, status = os.waitpid(child_pid, 0)

# child_pids = []
# parent(child_pids, '0.05')
# for child_pid in child_pids:
#     pid, status = os.waitpid(child_pid, 0)

print("Done.")