import os
import sys
import shutil
import json
import argparse
import h5py
import numpy as np
from progressbar import ProgressBar
from subprocess import call

def printout(flog, data):
    print(data)
    flog.write(data+'\n')

def check_mkdir(x):
    if os.path.exists(x):
        print('ERROR: folder %s exists! Please check and delete it!' % x)
        exit(1)
    else:
        os.mkdir(x)

def force_mkdir(x):
    if not os.path.exists(x):
        os.mkdir(x)

def force_mkdir_new(x):
    if not os.path.exists(x):
        os.mkdir(x)
    else:
        shutil.rmtree(x)
        os.mkdir(x)

def check_exist_dir(x):
    check_dir_exist(x)

def check_dir_exist(x):
    if not os.path.exists(x):
        print('ERROR: folder %s does not exists!' % x)
        exit(1)

def load_json(fn):
    with open(fn, 'r') as fin:
        return json.load(fin)

def save_h5(fn, pts, gt_label, gt_mask, gt_valid, gt_other_mask):
    fout = h5py.File(fn, 'w')
    fout.create_dataset('pts', data=pts, compression='gzip', compression_opts=4, dtype='float32')
    fout.create_dataset('gt_label', data=gt_label, compression='gzip', compression_opts=4, dtype='uint8')
    fout.create_dataset('gt_mask', data=gt_mask, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('gt_valid', data=gt_valid, compression='gzip', compression_opts=4, dtype='bool')
    fout.create_dataset('gt_other_mask', data=gt_other_mask, compression='gzip', compression_opts=4, dtype='bool')
    fout.close()

def load_h5(fn):
    with h5py.File(fn, 'r') as fin:
        pts = fin['pts'][:]
        label = fin['label'][:]
        return pts, label

def save_obj(obj, fname):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname):
    with open(fname + '.pkl', 'rb') as f:
        return pickle.load(f)