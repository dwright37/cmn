import sys
import os

from os import listdir
from os.path import isfile, join
import time
import json
import bisect
import numpy as np
import tensorflow as tf
from models import refgoog_attention_model

import util.io
from models.spatial_feat import spatial_feature_from_bbox
from util.refgoog_baseline_train.roi_data_reader import DataReader
from util import text_processing, im_processing
from models import refgoog_attention_model

T=20
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = refgoog_attention_model.fastrcnn_vgg_net.channel_mean
snapshot_file = './downloaded_models/refgoog_attbilstm_iter_150000_no_vgg.tfmodel'
num_vocab = 72704
embed_dim = 300 
lstm_dim = 1000

################################################################################
# The model
################################################################################

# Inputs
text_seq_batch = tf.placeholder(tf.int32, [T, None])
vis_feat = tf.placeholder(tf.float32, [None, 4096])
spatial_batch = tf.placeholder(tf.float32, [None, 5]) 

# Outputs
scores = refgoog_attention_model.refgoog_retrieval_baseline(vis_feat,
    spatial_batch, text_seq_batch, num_vocab, embed_dim, lstm_dim)

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Snapshot saver
snapshot_saver = tf.train.Saver()
snapshot_saver.restore(sess, snapshot_file)

def scoreImgOnQuery(text_seq_val, vfeat, sfeat):
   # Forward and Backward pass
    scores_val,((probs_obj1,probs_obj2,probs_rel),) = sess.run((scores, tf.get_collection('attention_probs')),
        feed_dict={
            text_seq_batch  : text_seq_val,
            vis_feat        : vfeat,
            spatial_batch   : sfeat
        })

    return np.max(scores_val, axis=1)[0],probs_obj1,probs_obj2,probs_rel

imdb_file = './exp-refgoog/data/imdb/imdb_val.npy'
image_dir = '/hdd/dustin/data/gref_train_1000_roifwdprop/'
reader = DataReader(imdb_file, vocab_file, im_mean, shuffle=True)

img_files = [join(image_dir, f) for f in listdir(image_dir) if isfile(join(image_dir, f)) and f[-4:] in '.npz']

vis_feats = []
spatial_feats = []

for im in img_files:
    with open(im, 'rb') as f:
        v = np.load(f)
        vis_feats.append(v['vis_feat'])
        spatial_feats.append(v['spatial_feature'])

mrr = []
results = []
for n in range(reader.num_batch):
    batch = reader.read_batch()
    #Get batch image activations
    name = os.path.basename(batch['im_path'])
    name = name[:name.rfind('.')]
    visfile = '/hdd/dustin/data/gref_val_roifwdprop/' + name + '.npz'
    with open(visfile, 'rb') as f:
        vcurr = np.load(f)
        vis_curr = vcurr['vis_feat']
        spatial_curr = vcurr['spatial_feature']
    
    ranks = []
    for (text_seq_val,q) in zip(batch['text_seq_batch'].T, batch['questions']):
        text_seq_val = text_seq_val[:, np.newaxis]
        curr_ranks = []
        for v,s in zip(vis_feats, spatial_feats):
            img_score,_,_,_ = scoreImgOnQuery(text_seq_val, v, s)
            bisect.insort(curr_ranks, -img_score)
        
        #Get the ground truth rank
        gt_score,probs_obj1,probs_obj2,probs_rel = scoreImgOnQuery(text_seq_val, vis_curr, spatial_curr)
        
        #Get the rank
        rank = bisect.bisect(curr_ranks, -gt_score)
        print(rank)
        mrr.append(1./(rank + 1))
        
        tval = [t for t in text_seq_val[:,0] if t != 0]
        pobj1 = [float(p) for p in probs_obj1[-len(tval):,0,0]]
        prel = [float(p) for p in probs_rel[-len(tval):,0,0]]
        pobj2 = [float(p) for p in probs_obj2[-len(tval):,0,0]]

        coll = {'rank': rank, 'question': q, 'im_path': batch['im_path'], 'probs_obj1':  pobj1, 'probs_obj2': pobj2, 'probs_rel': prel}
        results.append(coll)
    
        #Save results
        with open('results.json', 'w') as f:
            json.dump(results, f)

#Save results
with open('results.json', 'w') as f:
    json.dump(results, f)
print("MRR: ", sum(mrr) / len(mrr))
