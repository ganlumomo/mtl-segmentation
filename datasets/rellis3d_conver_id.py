import os
import sys
import numpy as np
from PIL import Image
import rellis3d_labels

RELLIS3D_DIR = '/home/cel/data/RELLIS_3D/Rellis-3D'


for seq in ['00000', '00001', '00002', '00003', '00004']:
    new_folder = os.path.join(RELLIS3D_DIR, seq, 'pylon_camera_node_train_id')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)


for dataset in ['train.lst', 'val.lst', 'test.lst']:
    file_path = os.path.join(RELLIS3D_DIR, dataset)
    new_dataset_file = os.path.join(RELLIS3D_DIR, 'trainid_'+dataset)
    output_file = open(new_dataset_file, "w")
    with open(file_path) as f:
        for line in f:
            split_line = line.split()
            rgb_img_path = os.path.join(RELLIS3D_DIR, split_line[0])
            label_id_path = os.path.join(RELLIS3D_DIR, split_line[1])
            new_file_path = split_line[1].replace('pylon_camera_node_label_id', 'pylon_camera_node_train_id')
            label_trainId_path = os.path.join(RELLIS3D_DIR, new_file_path)

            # read original label id image
            label_id = Image.open(label_id_path)
            label_id = np.array(label_id)

            # change label id to train id
            train_id = label_id.copy()
            for k, v in rellis3d_labels.label2trainid.items():
                train_id[label_id == k] = v
            train_id = Image.fromarray(train_id.astype(np.uint8))

            # save train id image
            train_id.save(label_trainId_path)
            
            # write to another text.lst file with new file path
            output_file.write(rgb_img_path+' '+new_file_path+'\n')
    output_file.close()
