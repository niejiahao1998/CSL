from __future__ import print_function, absolute_import
import os
import numpy as np
import random

# We add ir2rgb part because of our bi-directional evaluation protocol  ### date-2022/MAY/23
def process_query_sysu(data_path, mode = 'all', ir2rgb = 1, relabel=False): # ir2rgb=1 => IR->RGB; ir2rgb=2 => RGB->IR
    if mode== 'all':
        ### Modify test_mode simultaneously!!! (train_ext.py & testa.py)
        ### 1st line is default(IR -> Visible)
        ### 2nd line is Visible -> IR
        if ir2rgb == 1:
            ir_cameras = ['cam3','cam6']
        if ir2rgb == 2:
            ir_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        if ir2rgb == 1:
            ir_cameras = ['cam3','cam6']
        if ir2rgb == 2:
            ir_cameras = ['cam1','cam2']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

# We add ir2rgb part because of our bi-directional evaluation protocol 
def process_gallery_sysu(data_path, mode = 'all', trial = 0, ir2rgb = 1, relabel=False): # ir2rgb=1 => IR->RGB; ir2rgb=2 => RGB->IR
    
    random.seed(trial)
    
    if mode== 'all':
        if ir2rgb == 1:
            rgb_cameras = ['cam1','cam2','cam4','cam5']
        if ir2rgb == 2:
            rgb_cameras = ['cam3','cam6']
    elif mode =='indoor':
        if ir2rgb == 1:
            rgb_cameras = ['cam1','cam2']
        if ir2rgb == 2:
            rgb_cameras = ['cam3', 'cam6']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                # files_rgb.append(random.choice(new_files))
                try:
                    files_rgb.append(random.choice(new_files))
                except Exception as e:
                    print(new_files)

    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    
def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)

def process_test_ntu(data_dir, modal = 'visible'):

    test_id_path = os.path.join(data_dir, 'test.txt')
    test_id_list = open(test_id_path, 'r').read().splitlines()
    test_id = [s.split(' ')[0] for s in test_id_list]

    if modal=='visible':
        test_path = os.path.join(data_dir, 'RGB')
        
    elif modal=='thermal':
        test_path = os.path.join(data_dir, 'IR')

    file_image = []
    file_label = []
    for id in os.listdir(test_path):
        if id in test_id:
            id_path = os.path.join(test_path, id)
            for img in os.listdir(id_path):
                img_path = os.path.join(id_path,img)
                file_image.append(img_path)
                file_label.append(int(id)) 
    
    return file_image, np.array(file_label)