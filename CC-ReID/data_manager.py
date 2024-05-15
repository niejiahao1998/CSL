from __future__ import print_function, absolute_import
import os
import numpy as np
import random
import os.path as osp
import glob
import re


def process_test_prcc(img_dir):

    test_dir = osp.join(img_dir, 'rgb/test')
    pdirs = glob.glob(osp.join(test_dir, '*'))
    pdirs.sort()

    pid_container = set()
    for pdir in glob.glob(osp.join(test_dir, 'A', '*')):
        pid = int(osp.basename(pdir))
        pid_container.add(pid)
    pid_container = sorted(pid_container)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}

    query_img = []
    query_label = []
    gallery_img = []
    gallery_label = []

    for cam in ['A', 'C']:
        pdirs = glob.glob(osp.join(test_dir, cam, '*'))
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid = pid2label[pid]
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                if cam == 'A':
                    gallery_img.append(img_dir)
                    gallery_label.append(pid)
                elif cam == 'C':
                    query_img.append(img_dir)
                    query_label.append(pid)

    return query_img, np.array(query_label), gallery_img, np.array(gallery_label)

def process_test_ltcc(img_dir):

    test_id_file = osp.join(img_dir,'info','cloth-change_id_test.txt')
    with open(test_id_file, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids]

    query_dir = osp.join(img_dir, 'query')
    gallery_dir = osp.join(img_dir, 'test')
    query_img_paths = glob.glob(osp.join(query_dir, '*.png'))
    gallery_img_paths = glob.glob(osp.join(gallery_dir, '*.png'))
    query_img_paths.sort()
    gallery_img_paths.sort()
    pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
    pattern2 = re.compile(r'(\w+)_c')

    query_img = []
    query_id_label = []
    query_cam_label = []
    query_cloth_label = []
    for img_path in query_img_paths:
        pid, _, camid = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        if pid in ids:
            query_img.append(img_path)
            query_id_label.append(pid)
            query_cam_label.append(camid)
            query_cloth_label.append(clothes_id)
            
    gallery_img = []
    gallery_id_label = []
    gallery_cam_label = []
    gallery_cloth_label = []
    for img_path in gallery_img_paths:
        pid, _, camid = map(int, pattern1.search(img_path).groups())
        clothes_id = pattern2.search(img_path).group(1)
        if pid in ids:
            gallery_img.append(img_path)
            gallery_id_label.append(pid)
            gallery_cam_label.append(camid)
            gallery_cloth_label.append(clothes_id)           

    return query_img, np.array(query_id_label), np.array(query_cloth_label),\
        gallery_img, np.array(gallery_id_label), np.array(gallery_cloth_label)