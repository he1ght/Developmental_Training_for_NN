import subprocess
from glob import glob
fls = glob("tools/nmt_score.py")

test_type = ['commontest', 'newstest', 'nc-test']
nets_type = ['lstm', 'lstm_ce', 'trans', 'trans_ce']
embs_type = ['re', 'wv', 'bert']
data_name = {'commontest': 'test.en', 'newstest': 'newstest2014-fren-ppcd.en', 'nc-test': 'nc-test2007-ppcd.en'}

base_data_path = '/home/temp_use/data/'
base_pred_path = './predict/'

for fl in fls:
    for tt in test_type:
        for et in embs_type:
            for nt in nets_type:
                ref_path  = base_data_path + data_name[tt]
                pred_path = base_pred_path + tt + '_' + et + '_' + nt + '.txt'
                print("Test: {}, Network: {}, Embedding: {}".format(tt, nt, et))
                subprocess.call(['python', fl, ref_path, pred_path])