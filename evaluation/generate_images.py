#! /usr/bin/python
# -*- coding: utf8 -*-

import os, sys, math, pdb, pickle, random
import numpy as np
import scipy.misc
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

DATA_DIR = '../data/cub/test/'
softmax = None

BATCH_SIZE = 1
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=BATCH_SIZE)

def load_Gs(path):
    with open(path, 'rb') as f:
        _G, _D, Gs = pickle.load(f)
    return Gs
    
# -----生成测试集样本-----
def generate_img(Gs, test_texts):
    text_batch = []
    img_list = []
    
    testset_size = len(test_texts)
    text_num = test_texts[0].shape[0]
    
    for index in range(testset_size*text_num):
        if len(text_batch) < BATCH_SIZE - 1:
            text_batch.append(test_texts[index//text_num][index%text_num])
            continue
            
        text_batch.append(test_texts[index//text_num][index%text_num])
        text_batch_narray = np.asarray(text_batch)
        text_batch = []
        
        latents = np.random.random((BATCH_SIZE, Gs.input_shape[1]))
        outputs = Gs.run(latents, text_batch_narray, **synthesis_kwargs)
        
        outputs = np.minimum(np.maximum(outputs, 0.0), 255.0)
        for i in range(BATCH_SIZE):    
            img_list.append(outputs[i,:,:,:].astype(np.uint8))
            
        # if (index // 100) % 10 == 0:
        print('\r', 'Generating %d/%d' % (index+1, testset_size*text_num), end='')
    
    random.shuffle(img_list)
    return np.asarray(img_list)
    
    
        
def main():
    tflib.init_tf()
    # -----读取数据集-----    
    with open(os.path.join(DATA_DIR, 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        test_texts = u.load()
    
    # -----生成图像-----
    load_file = sys.argv[1]
    img_array = generate_img(load_Gs(load_file), test_texts)
    
    out_file = './generated_images.npy'
    np.save(out_file, img_array)
    

if __name__ == '__main__':
    main()