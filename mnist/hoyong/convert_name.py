import glob
import os
import sys

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
_path = "\\".join(sys.argv[1].split('\\'))
name_list = glob.glob(_path + '\\*.jpg')
for _name in name_list:
	_key = None
	for k in file_nickname_map.keys():
		if k in _name:
			_key = k
			break
	_new_name = _name.replace(_key, file_nickname_map[_key])
	os.rename(_name, _new_name)
# _raw_file = "\\".join(_raw_file)
# print(_raw_file)