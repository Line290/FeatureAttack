from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import time
import sys

import cifar10_input


class Feature_Attack:
    def __init__(self, model, config):
        self.model = model
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.step_size = config['step_size']
        self.rand = config['random_start']
        self.loss_func = config['loss_func']

        feat_dim = model.feat.get_shape()[-1]
        self.target_feats = tf.placeholder(tf.float32, shape=[None, feat_dim])
        if self.loss_func == 'cosine':
            normalized_input_feats = tf.nn.l2_normalize(self.model.feat, dim=1)
            normalized_target_feats = tf.nn.l2_normalize(self.target_feats, dim=1)
            cos_dist = 1. - tf.reduce_sum(normalized_input_feats * normalized_target_feats, axis=1) # cos: 1-1 not n-n
            loss = tf.reduce_mean(cos_dist)

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, target_imgs, sess):
        '''
        :param x_nat: single image needs to perturb, (1HWC)
        :param y:
        :param target_imgs: different targer images, (BHWC)
        :param sess:
        :return:
        '''
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255)  # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        target_feat_output = sess.run(self.model.feat, feed_dict={self.model.x_input: target_imgs})

        for i in range(self.num_steps):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.target_feats: target_feat_output})

            x = np.subtract(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255)  # ensure valid pixel range
        return x
    
if __name__ == '__main__':
    import json
    import sys
    import math


    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()
    else:
        print("Success load model: ", model_file)

    model = Model(mode='eval')
    attack = Feature_Attack(model, config)
    saver = tf.train.Saver()

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

    with tf.Session() as sess:
    # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        assert eval_batch_size == 1
        target_images_size = config['target_images_size']
        num_batches = num_eval_examples

        x_adv = [] # adv accumulator

        print('Iterating over {} batches'.format(num_batches))
        untarget_success_count = 0
        target_success_count = 0
        total = 0
        for ibatch in range(num_batches):
            start_time = time.time()

            # perturbed image
            x_batch = cifar.eval_data.xs[ibatch, :][None, ...]
            y_batch = cifar.eval_data.ys[ibatch]

            # target images
            batch_idx_list = {}
            other_label_test_idx = (cifar.eval_data.ys != y_batch)
            other_label_test_data = cifar.eval_data.xs[other_label_test_idx]
            other_label_test_label = cifar.eval_data.ys[other_label_test_idx]
            num_other_label_img = other_label_test_data.shape[0]
            # print(other_label_test_idx.shape, other_label_test_data.shape, other_label_test_label.shape)

            adv_idx = 0
            for i in range(1):
                num_target_imgs = 0
                target_img_list, target_label_list = [], []
                while num_target_imgs < target_images_size:
                    target_idx = random.randint(0, num_other_label_img - 1)
                    if target_idx in batch_idx_list:
                        continue
                    batch_idx_list[target_idx] = -1
                    target_label = other_label_test_label[target_idx]
                    target_input = other_label_test_data[target_idx][None, ...]
                    target_img_list.append(target_input)
                    target_label_list.append(target_label)
                    num_target_imgs += 1
                target_inputs = np.concatenate(target_img_list, 0)
                target_labels = np.array(target_label_list)
                x_batch_repeat = x_batch.repeat(target_images_size, 0)
                y_batch_repeat = y_batch.repeat(target_images_size)
                # print(x_batch_repeat.shape, y_batch_repeat)
                # print(target_inputs.shape, target_labels)


                x_batch_adv = attack.perturb(x_batch_repeat, target_inputs, sess)

                preds = sess.run(model.predictions, feed_dict={model.x_input: x_batch_adv})
                # print(preds)
                not_correct_idices = (preds.reshape(-1) != y_batch_repeat.reshape(-1))
                not_correct_num = not_correct_idices.sum()
                attack_success_num = (preds.reshape(-1) == target_labels.reshape(-1)).sum()
                # At least one misclassified
                if not_correct_num != 0:
                    untarget_success_count += 1
                    if attack_success_num != 0:
                        target_success_count += 1
                    adv_idx = np.argwhere(not_correct_idices == True).reshape(-1)[0]
                    break

            total += 1
            duration = time.time() - start_time
            x_adv.append(x_batch_adv[adv_idx][None, ...])

            if ibatch % 1 == 0:
                print(
                    "step %d, duration %.2f, aver untargeted attack success %.2f, aver targeted attack success %.2f"
                    % (ibatch, duration, 100. * untarget_success_count / total, 100. * target_success_count / total))
                sys.stdout.flush()
        acc = 100. * untarget_success_count / total
        print('Val acc:', acc)
        print('Storing examples')
        path = config['store_adv_path']
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))
