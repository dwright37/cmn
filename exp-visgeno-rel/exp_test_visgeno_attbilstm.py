from __future__ import absolute_import, division, print_function

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # using GPU 0

import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform

from models import visgeno_attention_model, spatial_feat, fastrcnn_vgg_net
from util.visgeno_rel_train.rel_data_reader import DataReader
from util import loss, eval_tools, text_processing
import util.io
import ipdb

################################################################################
# Parameters
################################################################################

# Model Params
T = 20
num_vocab = 72704
embed_dim = 300
lstm_dim = 1000

# Data Params
imdb_file = './exp-visgeno-rel/data/imdb/imdb_tst.npy'
vocab_file = './word_embedding/vocabulary_72700.txt'
im_mean = visgeno_attention_model.fastrcnn_vgg_net.channel_mean

# Snapshot Params
model_file = './downloaded_models/visgeno_attbilstm_strong_iter_360000.tfmodel'
#model_file = './downloaded_models/visgeno_attbilstm_weak_iter_360000.tfmodel'

result_file = './exp-visgeno-rel/results/visgeno_attbilstm_strong_iter_360000_tst_1.txt'
result_json = './exp-visgeno-rel/results/visgeno_attbilstm_strong_iter_360000_tst_1.json'
################################################################################
# Network
################################################################################

im_batch = tf.placeholder(tf.float32, [1, None, None, 3])
bbox_batch = tf.placeholder(tf.float32, [None, 5])
spatial_batch = tf.placeholder(tf.float32, [None, 5])
text_seq_batch = tf.placeholder(tf.int32, [T, None])

scores, probs_obj1, probs_obj2, probs_rel = visgeno_attention_model.visgeno_attbilstm_net(im_batch, bbox_batch, spatial_batch,
    text_seq_batch, num_vocab, embed_dim, lstm_dim, False, False)

np.random.seed(3)
reader = DataReader(imdb_file, vocab_file, im_mean, shuffle=False, max_bbox_num=10000, max_rel_num=10000)

################################################################################
# Snapshot and log
################################################################################

# Snapshot saver
snapshot_saver = tf.train.Saver()

# Start Session
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

# Run Initialization operations
snapshot_saver.restore(sess, model_file)

K = 10
top_x_correct_count = np.zeros(K)  # compute up to top-K accuracy
att_sum_obj1 = 0
att_sum_obj2 = 0
att_sum_rel = 0
total = 0

################################################################################
# Optimization loop
################################################################################

# Run optimization
for n_iter in range(reader.num_batch):
    batch = reader.read_batch()
    print('\tthis batch: N_lang = %d, N_bbox = %d' %
          (batch['expr_obj1_batch'].shape[1], batch['bbox_batch'].shape[0]))

    # Forward and Backward pass
    scores_val, ((probs_obj1, probs_obj2, probs_rel),) = sess.run((scores,
            tf.get_collection("attention_probs")),
        feed_dict={
            im_batch            : batch['im_batch'],
            bbox_batch          : batch['bbox_batch'],
            spatial_batch       : batch['spatial_batch'],
            text_seq_batch      : batch['text_seq_batch']
        })

    N_batch, N_box, _, _ = scores_val.shape

    # scores_val has shape [N_batch, N_box, N_box, 1]
    scores_obj1 = np.max(scores_val.reshape((N_batch, N_box, N_box)), axis=2)
    # prediction_box_ids has shape [N_batch, K] containing indices
    prediction_box_ids = np.argsort(-scores_obj1, axis=1)[:, :K]  # minus to sort in descending order
    # labels has shape [N_batch, 1] containing indices
    labels = batch['label_weak_batch'].reshape((N_batch, 1))

    is_matched = (prediction_box_ids == labels).astype(np.float32)
    is_matched_cumsum = np.cumsum(is_matched, axis=1)
    matched_ids_count = np.sum(is_matched_cumsum, axis=0)
    top_x_correct_count[:N_box] += matched_ids_count
    top_x_correct_count[N_box:] += N_batch
    total += N_batch

    ipdb.set_trace()

    # save json
    if n_iter == 0:
        eval_output_json = []
    for n_question in range(N_batch):
        result = {
            "image_path": batch["im_path"],
            "predicted_bounding_boxes": batch['bbox_orig'][prediction_box_ids[n_question,:]],
            "refexp": batch["questions"][n_question],
            "obj1_prob": probs_obj1[:,n_question,:],
            "obj2_prob": probs_obj2[:,n_question,:],
            "rel_prob": probs_rel[:,n_question,:],
            "ground_truth": batch['bbox_orig'][labels[n_question,:]]
        }
        eval_output_json.append(result)
    if n_iter == 0: # check if save passes..
        util.io.save_json(eval_output_json, result_json)
    if n_iter == reader.num_batch - 1:
        util.io.save_json(eval_output_json, result_json)
        print('evaluation output file saved to %s' % result_json)
    

with open(result_file, 'w') as f:
    for k in range(K):
        f.write('recall at %d: %f (= %d / %d)\n' %
                (k+1, top_x_correct_count[k]/total, top_x_correct_count[k], total))
print('Testing results saved to %s' % result_file)
