"""Loss functions."""

import pdb
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]
    
#----------------------------------------------------------------------------
# Mutual information loss function

def compute_mi_loss(mi_fc2, labels, mi_lambda=1.0, fix_std=True):
    label_size = labels.get_shape()[1].value
    mean_contig = mi_fc2[:, 0:label_size]
    
    if fix_std:
        std_contig = tf.ones_like(mean_contig)
    else:
        std_contig = tf.sqrt(tf.exp(mi_fc2[:, label_size:label_size*2]))

    epsilon = (labels - mean_contig) / (std_contig + 1e-6)
    mi_l1 = tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - tf.log(std_contig + 1e-6) - 0.5 * tf.square(epsilon), reduction_indices=1)
    
    mi_loss = mi_lambda * mi_l1
    return mi_loss

#----------------------------------------------------------------------------
# Loss functions used by the paper

def G_logistic_nonsaturating(G, D, opt, training_set, minibatch_size, labels):
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    
    fake_scores_out, fake_mi_fc2 = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = fp32(fake_scores_out)
    
    dlatents, mi_fc2 = G.components.mapping.get_output_for(latents, labels, is_training=True)
    cw_mi_loss = compute_mi_loss(mi_fc2, labels)
    
    GAN_loss = tf.nn.softplus(-fake_scores_out)
    mi_loss = compute_mi_loss(fake_mi_fc2, labels)
    loss = GAN_loss - mi_loss - cw_mi_loss

    return loss

def D_logistic_simplegp(G, D, opt, training_set, minibatch_size, reals, wrongs, labels, r1_gamma=10.0, r2_gamma=0.0, r3_gamma=0.0): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    
    real_scores_out, real_mi_fc2 = D.get_output_for(reals, labels, is_training=True)
    real_scores_out = fp32(real_scores_out)
    wrong_scores_out, wrong_mi_fc2 = D.get_output_for(wrongs, labels, is_training=True)
    wrong_scores_out = fp32(wrong_scores_out)
    fake_scores_out, fake_mi_fc2 = D.get_output_for(fake_images_out, labels, is_training=True)
    fake_scores_out = fp32(fake_scores_out)
    
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    wrong_scores_out = autosummary('Loss/scores/wrong', wrong_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    GAN_loss = tf.nn.softplus(fake_scores_out) + tf.nn.softplus(-real_scores_out) + tf.nn.softplus(wrong_scores_out)

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out))
            real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals])[0]))
            r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
            r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        GAN_loss += r1_penalty * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out))
            fake_grads = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss, [fake_images_out])[0]))
            r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3])
            r2_penalty = autosummary('Loss/r2_penalty', r2_penalty)
        GAN_loss += r2_penalty * (r2_gamma * 0.5)
        
    if r3_gamma != 0.0:
        with tf.name_scope('R3Penalty'):
            wrong_loss = opt.apply_loss_scaling(tf.reduce_sum(wrong_scores_out))
            wrong_grads = opt.undo_loss_scaling(fp32(tf.gradients(wrong_loss, [wrong_images_out])[0]))
            r3_penalty = tf.reduce_sum(tf.square(wrong_grads), axis=[1,2,3])
            r3_penalty = autosummary('Loss/r3_penalty', r3_penalty)
        GAN_loss += r3_penalty * (r3_gamma * 0.5)
        
    mi_loss = compute_mi_loss(fake_mi_fc2, labels)
    loss = GAN_loss - mi_loss
    return loss

#----------------------------------------------------------------------------
