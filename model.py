from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev

@tf.RegisterGradient("MeanGrad")
def mean_grad(op, grad):
    return grad / cluster.tower_space.task.num()

def _allgather_grad(op, grad):
    """Gradient for allgather op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    grad = HorovodVariableMgr.allreduce(grad, average=False)

    x = op.inputs[0]
    d0 = x.get_shape().as_list()[0]
    d = tf.convert_to_tensor([d0], dtype=tf.int32)

    s = HorovodVariableMgr.num_workers()
    d = tf.reshape(HorovodVariableMgr.allgather(d), [s])

    splits = tf.split(grad, num_or_size_splits=d, axis=0)
    return splits[HorovodVariableMgr.rank()]

class Model(BasicCNN):

    def __init__(self, train_mode, branch_scheduler,
                total_blocks=4,
                layers_in_block=(5,10,20,5),
                densenet_k_block=(12,12,12,12),
                res_dim = (64,128,256,512)):
        super().__init__(train_mode)
        
        self.train_mode = train_mode
        self.branch_scheduler = branch_scheduler
        self.total_blocks = total_blocks 
        self.layers_in_block = layers_in_block
        self.densenet_k_block = densenet_k_block
        self.first_output_features = 32
        self.res_dim = res_dim
        self.dpn_width = 0
        self.dense_cprs_rate = 0.25
        self.se_ratio = 8
        self.act_type = 'prelu'
        self.ECCV_domains = ['bright_male', 'bright_female', 'dark_male', 'dark_female']
        self.domains = ['Indian_train', 'SEA_train', 'black_train', 'black2', 'CFP']
        self.domain_feat = {}
        for d in self.ECCV_domains:
            mean_feat_path = 'mean_feat_path'+d+'.fea'
            mean_feat = []
            for line in open(mean_feat_path):
                mean_feat.append(np.array(line.rstrip().split()).astype('float32'))
            mean_feat = np.array(mean_feat)
            mean_feat /= np.linalg.norm(mean_feat, axis=1, keepdims=True)
            self.domain_feat[d] = mean_feat.T
        for d in self.domains:
            mean_feat_path = 'mean_feat_path'+d+'.fea'
            mean_feat = []
            for line in open(mean_feat_path):
                mean_feat.append(np.array(line.rstrip().split()).astype('float32'))
            mean_feat = np.array(mean_feat)
            mean_feat /= np.linalg.norm(mean_feat, axis=1, keepdims=True)
            self.domain_feat[d] = mean_feat.T

    def bn_relu_conv(self, x, chn_out, ksize, stride=1, pad=True, scope=None):
        with tf.variable_scope(scope) as scope:
            x = self.batch_norm(x, scope=scope)
            x = tf.nn.relu(x)
            x = self.convolve(x,
                              channel_out=chn_out,
                              ksize=ksize,
                              stride=stride,
                              withpad=pad,
                              scope=scope)

            return x

    def act(self, x, act_type='prelu', scope=None):
        with tf.variable_scope(scope) as scope:
            if act_type=='prelu':
                x = tf.nn.leaky_relu(x)
            elif act_type=='relu':
                x = tf.nn.relu(x)
            return x

    def res_unit2(self, inputs,  num_filter, stride=1, bottle_neck=True, se_set=False,scope=None):
        unit_channel = inputs.get_shape().as_list()[-1]

        if bottle_neck:
            x = self.batch_norm(inputs, scope='bn1')
            x = self.convolve(x, channel_out=int(num_filter*0.25), ksize=1, stride=1, scope='conv1')
            x = self.batch_norm(x, scope='bn2')
            x = self.act(x, act_type=self.act_type, scope='act1')
            x = self.convolve(x, channel_out=int(num_filter*0.25), ksize=3, stride=stride, scope='conv2')
            x = self.batch_norm(x, scope='bn3')
            x = self.act(x, act_type=self.act_type, scope='act2')
            x = self.convolve(x, channel_out=num_filter, ksize=1, stride=1, scope='conv3')
            x = self.batch_norm(x, scope='bn4')
        
            if stride != 1 or num_filter != unit_channel:
                conv1sc = self.convolve(inputs, channel_out=num_filter, ksize=1, stride=stride, scope='conv1sc')
                shortcut = self.batch_norm(conv1sc, scope='bn_sc')
            else:
                shortcut = inputs
        
        else:
            x = self.batch_norm(inputs, scope='bn1')
            x = self.convolve(x, channel_out=num_filter, ksize=3, stride=1, scope='conv1')
            x = self.batch_norm(x, scope='bn2')
            x = self.act(x, act_type=self.act_type, scope='act1')
            x = self.convolve(x, channel_out=num_filter, ksize=3, stride=stride, scope='conv2')
            x = self.batch_norm(x, scope='bn3')
            
            if stride != 1 or num_filter != unit_channel:
                #print('@@@@@@@@@@@')
                conv1sc = self.convolve(inputs, channel_out=num_filter, ksize=1, stride=stride, scope='conv1sc')
                shortcut = self.batch_norm(conv1sc, scope='bn_sc')               
            else:
                shortcut = inputs
                #print ('!!!!!!')

        if se_set:
            x = self.se_block(x, ratio=self.se_ratio, scope='res_unit_se')

        residual = x + shortcut
        mem_save.checkpoint(residual)
        return residual #x + shortcut   
   
    def se_block(self, input_x, ratio, scope):
        out_dim = int(input_x.get_shape()[-1])
        with tf.variable_scope(scope) :
            squeeze = tf.reduce_mean(input_x, [1,2])
            excitation = self.fc_unit(squeeze, out_units=max(out_dim//ratio,16), scope='se_fu1')
            excitation = self.fc_unit(excitation, out_units=out_dim, scope='se_fu2')
            excitation = tf.nn.sigmoid(excitation)
            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation
            return scale
 
    def reduction(self, inputs, chnl_inc, stride, pad=True):
        x = self.batch_norm(inputs, scope='redu_bn')
        x = tf.nn.relu(x)
        x = self.convolve(x,
                    channel_out=chnl_inc,
                    ksize=3,
                    stride=stride,
                    withpad=pad,
                    scope='redu_conv')

        pooled = self.max_pool(inputs,
                                   ksize=3,
                                   stride=stride,
                                   withpad=pad)
        merged = tf.concat(values=[x, pooled], axis=3)

        return merged
    
    def mixed_pooling(self, inputs, chnl_inc, stride, pad=True):
        
        x = self.bn_relu_conv(inputs,
                    chn_out=chnl_inc,
                    ksize=3,
                    stride=stride,
                    pad=pad,
                    scope='down_conv')

        pooled = self.max_pool(inputs,
                                   ksize=3,
                                   stride=stride,
                                   withpad=pad)
        
        scale_var = tf.get_variable('mixed_pool_coef',
                                        shape=[1],
                                        dtype=tf.float32,
                                        trainable=True,
                                        initializer=tf.constant_initializer(0.5))
        
        merged = scale_var*x + (1-scale_var)*pooled
        
        return merged

    def res_stack2(self, x, unit_num, num_channel,se_set=False):
        vec = [x]
        for n in range(unit_num):
            with tf.variable_scope('block{0}'.format(n)):
                x = self.res_unit2(x, num_channel, se_set=se_set)
        return x

    def fc_unit(self, inputs, out_units, scope='fu_'):
        with tf.variable_scope(scope):
            x = self.batch_norm(inputs, scope='bn1')
            x = tf.nn.relu(x)
            x = self.fullconnect(x, num_units_out=out_units, scope='fc')
            x = self.batch_norm(x, scope='bn2')
        return x

    def fc_norm_unit_distributed(self, inputs, out_units, name):
        out_num = out_units
        stddev = 1.0 / tf.sqrt(tf.cast(out_units, tf.float32))

        shard_devices = ['/job:{}/task:{}/gpu:{}'.format('worker', task_index, 0)
                             for task_index in cluster.tower_space.task_ids()]
        logger.log('DEBUG: FC weight shards on devices:')
        for device in shard_devices:
            logger.log_with_prefix(device)

        class choose_device(object):
            def __init__(self):
                self.i = -1
                self._var_op = ["Variable", "VariableV2", "VarHandleOp"]

            def __call__(self, op):
                node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
                if node_def.op in self._var_op:
                    self.i = self.i + 1
                    return shard_devices[self.i]
                else:
                    return ''

        with tf.variable_scope('dist_model',
                                partitioner=tf.fixed_size_partitioner(num_shards=cluster.tower_space.task.num(), axis=1)):
            with tf.device(choose_device()):
                if 'ECCV' in name:
                    cur_domain = None
                    for d in self.ECCV_domains:
                        if d in name:
                            cur_domain = d
                            break
                    weights = tf.get_variable(name=name, shape=(inputs.get_shape().as_list()[-1], out_num),
                                    #   initializer=tf.random_uniform_initializer(minval=-stddev, maxval=stddev,
                                    #                                             dtype=tf.float32),
                                      initializer=self.domain_feat[cur_domain],
                                      dtype=tf.float32,
                                      regularizer=None,
                                      trainable=True)
                elif not 'ms1m' in name:
                    cur_domain = None
                    for d in self.domains:
                        if d in name:
                            cur_domain = d
                            break
                    weights = tf.get_variable(name=name, shape=(inputs.get_shape().as_list()[-1], out_num),
                                    #   initializer=tf.random_uniform_initializer(minval=-stddev, maxval=stddev,
                                    #                                             dtype=tf.float32),
                                      initializer=self.domain_feat[cur_domain],
                                      dtype=tf.float32,
                                      regularizer=None,
                                      trainable=True)
                else:
                    weights = tf.get_variable(name=name, shape=(inputs.get_shape().as_list()[-1], out_num),
                                    #   initializer=tf.random_uniform_initializer(minval=-stddev, maxval=stddev,
                                    #                                             dtype=tf.float32),
                                      initializer=tf.contrib.layers.xavier_initializer(),
                                      dtype=tf.float32,
                                      regularizer=None,
                                      trainable=True)

        logger.log('DEBUG: FC weight shards:')
        weights = [w for w in weights]
        for w in weights:
            tf.add_to_collection('local_update_partitioned_var_slices', w)
            tf.add_to_collection('shared_global_vars', w)
            logger.log_with_prefix('{} -> {}'.format(w, w.device))

        shard_index = cluster.tower_space.task.index()
        shard_sizes = [w.get_shape().as_list()[1] for w in weights]
        shard_ranges = [0]
        for s in shard_sizes:
            shard_ranges.append(shard_ranges[-1]+s)
        shard_range = shard_ranges[shard_index:shard_index+2]
        logger.log('DEBUG: shard_ranges = {}, shard_range = {}'.format(shard_ranges, shard_range))

        weights = weights[shard_index]
        regularizer=tf.contrib.layers.l2_regularizer(0.001)
        if regularizer:
            with tf.colocate_with(weights):
                weights_name = weights.name
                if weights_name.endswith(':0'):
                    weights_name = weights_name[:-len(':0')]
                with tf.name_scope(weights_name + "/Regularizer/"):
                    loss = regularizer(weights)
                    if loss is not None:
                        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, loss)

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "MeanGrad"}):
            weights = tf.identity(weights, name="Identity")


        weights = tf.nn.l2_normalize(weights, dim=0)
        embedding = inputs
        embedding = tf.nn.l2_normalize(embedding, dim=1)

        cos_t = tf.matmul(embedding, weights, name='cos_t')
        return cos_t, shard_range[0]

    def inference(self, all_sampels, all_labels, batch_sizes, pair_batch_sizes, num_classes, data_names, pair_flag):
        with tf.variable_scope('initial'):
            x = self.convolve(all_sampels,
                              channel_out=64,
                              ksize=7,
                              stride=2,
                              scope='stem_conv')
            x = self.max_pool(x, ksize=3, stride=2)

        # shape: 62*62*32 -> 62*62*64
        with tf.variable_scope('stage1'):
            x = self.res_unit2(x, 256)
            x = self.res_stack2(x, 2,256)  #2, 5

        # shape: 62*62*64 -> 31*31*128
        with tf.variable_scope('stage2'):
            x = self.res_unit2(x, 512, stride=2)
            x = self.res_stack2(x, 3, 512)  #3, 10

        # shape: 31*31*128 -> 15*15*256
        with tf.variable_scope('stage3'):
            x = self.res_unit2(x, 1024, stride=2)
            x = self.res_stack2(x, 22, 1024)  #6, 15, 40

        # shape: 15*15*256 -> 8*8*384
        with tf.variable_scope('stage4'):
            x = self.res_unit2(x, 2048, stride=2,se_set=True)
            x = self.res_stack2(x, 3, 2048, se_set=True)  #2,  5
        
        with tf.variable_scope('stage5'):
            feature_total = int(x.get_shape()[-1]) * int(x.get_shape()[-2]) * int(x.get_shape()[-3])
            x = tf.reshape(x, [-1, feature_total])
        dimension = 512 # feature dimension
        net_out = self.fc_unit(x, dimension) 

        all_batch_sizes = batch_sizes + pair_batch_sizes + pair_batch_sizes
        worker_num = cluster.tower_space.task.num()

        net_out_split = tf.split(net_out, num_or_size_splits=all_batch_sizes)
        branch_outs = []
        branch_labels = []
        pair1_outs = []
        pair2_outs = []
        for i, batch_size in enumerate(batch_sizes):
            # logger.log('batch_size: ' + str(batch_size))
            branch_outs.append(HorovodVariableMgr.allgather(net_out_split[i]))
            branch_outs[-1].set_shape([batch_size * worker_num, 512])
            branch_outs[-1] = tf.nn.l2_normalize(branch_outs[-1], axis=1)
            branch_labels.append(HorovodVariableMgr.allgather(all_labels[i]))
            branch_labels[-1].set_shape([batch_size * worker_num,])
        norm_branch_num = len(batch_sizes)
        pair_branch_num = len(pair_batch_sizes)
        for i, batch_size in enumerate(pair_batch_sizes):
            pair1_outs.append(HorovodVariableMgr.allgather(net_out_split[i+norm_branch_num]))
            pair1_outs[-1].set_shape([batch_size * worker_num, 512])
            pair1_outs[-1] = tf.nn.l2_normalize(pair1_outs[-1], axis=1)
            pair2_outs.append(HorovodVariableMgr.allgather(net_out_split[i+norm_branch_num+pair_branch_num]))
            pair2_outs[-1].set_shape([batch_size * worker_num, 512])
            pair2_outs[-1] = tf.nn.l2_normalize(pair2_outs[-1], axis=1)


        branch_logits = []
        weight_range_bases = []
        
        with tf.variable_scope('logits'):
            num_classes = [num_classes[key] for key in data_names]
            for i in range(norm_branch_num):
                name = data_names[i] + '/fu_fc'
                logit, weight_range_base = self.fc_norm_unit_distributed(branch_outs[i], num_classes[i], name)
                branch_logits.append(logit)
                weight_range_bases.append(weight_range_base)
        return weight_range_bases, branch_logits, branch_labels, branch_outs, pair1_outs, pair2_outs


    def _arcface_sftmax_loss_distributed(self, logits, labels, weight_range, m=0.5, s=64.0):
        def log(x, name):
            return x

        out_num = tf.shape(logits)[1]

        cos_t = logits

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = sin_m * m

        use_threshold = True
        prevent_exp_overflow = True
        prevent_log_overflow = True

        with tf.variable_scope('arcface_loss'):
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            if use_threshold:
                threshold = math.cos(math.pi - m)
                cond_v = cos_t - threshold
                cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

                keep_val = s*(cos_t - mm)
                cos_mt_temp = tf.where(cond, cos_mt, keep_val)
            else:
                cos_mt_temp = cos_mt

            mask = tf.one_hot(labels - weight_range, depth=out_num, name='one_hot_mask')
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

            output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
            
            f_k = output
            time.sleep(1)
            now = int(time.time())
            g = tf.get_default_graph()
            with g.gradient_override_map({"Identity": "ArcfaceGrad_{}".format(now)}):
                f_k_identity = tf.identity(f_k, name="Identity")

            f_k = f_k_identity + tf.stop_gradient(f_k - f_k_identity)

            if prevent_exp_overflow:
                #allreduce max
                max_f_k = tf.reduce_max(f_k, axis=1)
                max_f_k_all = HorovodVariableMgr.allgather(max_f_k)
                max_f_k_reshaped = tf.reshape(max_f_k_all, [cluster.tower_space.task.num(), -1])
                max_f_k = tf.reduce_max(max_f_k_reshaped, axis=0)
                max_f_k = tf.expand_dims(max_f_k, 1)
                exp_f_k = tf.exp(f_k - max_f_k)
            else:
                exp_f_k = tf.exp(f_k)
            exp_f_k = tf.cast(exp_f_k, tf.float64)
            exp_f_k = log(exp_f_k, 'exp_f_k')

            sum_exp_f_k = tf.reduce_sum(exp_f_k, 1)

            all_sum_exp_f_k = tf.stop_gradient(HorovodVariableMgr.allreduce(sum_exp_f_k, average=False))
            sum_exp_f_k = all_sum_exp_f_k
            sum_exp_f_k = log(sum_exp_f_k, 'sum_exp_f_k')

            log_sum_exp_f_k = tf.expand_dims(tf.log(sum_exp_f_k), 1)
            log_sum_exp_f_k = tf.cast(log_sum_exp_f_k, tf.float32)
            log_p_k = (f_k - max_f_k) - log_sum_exp_f_k

            @tf.RegisterGradient("ArcfaceGrad_{}".format(now))
            def arcface_grad(op, grad):
                _res = tf.exp(log_p_k) - mask
                _res = _res / log_p_k.get_shape().as_list()[0] * cluster.tower_space.task.num()
                return _res

            L_i = tf.reduce_sum(mask * (-log_p_k), 1)
            L_i = log(L_i, 'L_i')

            L = tf.reduce_mean(L_i)

            sum_L = HorovodVariableMgr.allreduce(L, average=False)
            tf.add_to_collection('loss_value', sum_L)

            return sum_L
    
    # TODO loss weight config
    def _pair_loss(self, branch_feats, branch_labels,  pair_left_feats, pair_right_feats, add_black_feats, add_black_labels, scope=''):
        assert len(branch_feats) == len(branch_labels)
        assert len(pair_right_feats) == len(pair_left_feats)

        def cal_local_prob(sim_mat):
            bins = tf.convert_to_tensor(np.arange(0, 1, 0.01), dtype=tf.float32)
            bins = tf.expand_dims(bins, 0)  # shape 1*100
            sim_mat = tf.expand_dims(sim_mat, 1)  # shape: bz * 1
            sim_mat_p = tf.matmul(sim_mat, tf.ones_like(bins), transpose_b=False) - tf.matmul(tf.ones_like(sim_mat), bins, transpose_b=False)  # shape: bz*100
            sim_mat_p = tf.reduce_sum(tf.exp(-0.5 * tf.square(sim_mat_p / 0.1)), axis=0, keep_dims=True)  # shape: 1*100
            p_sum = tf.reduce_sum(sim_mat_p, axis=1, keep_dims=False)  # shape : 1
            sim_mat_p_normed = sim_mat_p / p_sum  # shape: 1*100
            return sim_mat_p_normed

        with tf.variable_scope(scope):
            topk_neg_sims = []
            topk_neg_probs = []
            for i, (branch_feat, branch_label) in enumerate(zip(branch_feats, branch_labels)):
                sim_mat = tf.matmul(branch_feat, branch_feat, transpose_a=False, transpose_b=True)
                sim_mask = tf.cast(tf.not_equal(tf.expand_dims(branch_label, 1) - tf.expand_dims(branch_label, 0), 0), dtype=tf.float32)
                sim_mat = sim_mat * sim_mask - (1.0 - sim_mask)
                sim_mat = tf.reshape(sim_mat, [-1])
                topk_neg_sim = tf.nn.top_k(sim_mat, k=100)[0] # or 1000?
                topk_neg_sims.append(topk_neg_sim)
                topk_neg_probs.append(cal_local_prob(topk_neg_sim))
                
            pos_sims = []
            pos_probs = []
            for i, (pair_left_feat, pair_right_feat) in enumerate(zip(pair_left_feats, pair_right_feats)):
                # sim_mat = tf.matmul(pair_left_feat, pair_right_feat, transpose_a=False, transpose_b=True)
                # sim_mat = tf.matrix_diag_part(sim_mat)
                sim_mat = tf.reduce_sum(tf.multiply(pair_left_feat, pair_right_feat), axis=1)
                sim_mat = tf.reshape(sim_mat, [-1])
                # sim_mat = tf.nn.top_k(sim_mat, k=256)[0][208:252] # 256 in total
                # indices = tf.logical_and(tf.less(sim_mat, 0.6), tf.greater_equal(sim_mat, 0.2))
                indices = tf.less(sim_mat, 0.6)
                selected_sim_mat = tf.gather_nd(sim_mat, tf.where(indices))
                sim_mat = tf.cond(tf.size(selected_sim_mat) > 0, lambda: selected_sim_mat, lambda: sim_mat) 
                pos_sims.append(sim_mat)
                pos_probs.append(cal_local_prob(sim_mat))

            for i, (topk_neg_prob, pos_prob) in enumerate(zip(topk_neg_probs, pos_probs)):
                tf.summary.histogram(str(i)+'pos_sims', pos_sims[i])
                tf.summary.histogram(str(i)+'neg_sims', topk_neg_sims[i])
                entropy_loss = - tf.reduce_sum(tf.multiply(topk_neg_prob, tf.log(topk_neg_prob + 1e-9)))
                tf.add_to_collection('entropy_losses', entropy_loss)
                tf.add_to_collection('loss_value', 0.01 * entropy_loss) # TODO
                entropy_loss = - tf.reduce_sum(tf.multiply(pos_prob, tf.log(pos_prob + 1e-9)))
                tf.add_to_collection('entropy_losses',  entropy_loss) 
                tf.add_to_collection('loss_value', 0.01 * entropy_loss) # TODO
            
            for pair in [(0,2),(1,3),(0,1),(2,3)]:
                kl_loss = tf.reduce_sum(tf.multiply(topk_neg_probs[pair[0]] + 1e-9, tf.log(tf.div(topk_neg_probs[pair[0]] + 1e-9, topk_neg_probs[pair[1]] + 1e-9) + 1e-9)))
                tf.add_to_collection('kl_losses', kl_loss)
                tf.add_to_collection('loss_value', 0.1 * kl_loss) # TODO
                kl_loss = tf.reduce_sum(tf.multiply(pos_probs[pair[0]] + 1e-9, tf.log(tf.div(pos_probs[pair[0]] + 1e-9, pos_probs[pair[1]] + 1e-9) + 1e-9)))
                tf.add_to_collection('kl_losses', kl_loss)
                tf.add_to_collection('loss_value', 0.1 * kl_loss) # TODO
                # overlap_loss

            for i in range(4):
                order_loss = - (tf.reduce_mean(pos_sims[i]) - tf.reduce_mean(topk_neg_sims[i]))
                tf.add_to_collection('order_losses', order_loss)
                tf.add_to_collection('loss_value', 0.1 * order_loss) # TODO

            for i, (branch_feat, branch_label) in enumerate(zip(add_black_feats, add_black_labels)):
                sim_mat = tf.matmul(branch_feat, branch_feat, transpose_a=False, transpose_b=True)
                sim_mask = tf.cast(tf.not_equal(tf.expand_dims(branch_label, 1) - tf.expand_dims(branch_label, 0), 0), dtype=tf.float32)
                sim_mat = sim_mat * sim_mask - (1.0 - sim_mask)
                sim_mat = tf.reshape(sim_mat, [-1])
                add_topk_neg_sim = tf.nn.top_k(sim_mat, k=100)[0] # or 1000?
                add_topk_neg_prob = cal_local_prob(add_topk_neg_sim)
                tf.summary.histogram(str(i)+'add_neg_sims', add_topk_neg_sim)
                entropy_loss = - tf.reduce_sum(tf.multiply(topk_neg_prob, tf.log(topk_neg_prob + 1e-9)))
                tf.add_to_collection('entropy_losses', entropy_loss)
                tf.add_to_collection('loss_value', 0.01 * entropy_loss) # TODO
                order_loss = tf.reduce_mean(add_topk_neg_sim)
                tf.add_to_collection('order_losses', order_loss)
                tf.add_to_collection('loss_value', 0.1 * order_loss) # TODO
            
                


    def loss(self, images, labels, pair_images, pair_labels, num_classes, data_names, loss_scale=1.0, scope=None):
        pair_flag = True if len(pair_labels) > 0 else False
        
        # generate branch batch_size
        batch_sizes = [x.get_shape()[0].value for x in images]

        # generate pair batch_size
        if pair_flag:
            pair_batch_sizes = [x.get_shape()[0].value for x in pair_images[::2]] 
        else:
            pair_batch_sizes = []
        
        # generate labels
        if pair_flag:
            all_labels = labels + pair_labels + pair_labels
        else:
            all_labels = labels
        
        # generate element score
        samples = tf.concat(images, 0)
        if pair_flag:
            left_pair_images = tf.concat(pair_images[::2], 0)
            right_pair_images = tf.concat(pair_images[1::2], 0)
            all_samples = tf.concat([samples, left_pair_images, right_pair_images], 0)
        else:
            all_samples = samples

        # inference
        weight_range_bases, branch_logits, branch_labels, branch_outs, pair1_outs, pair2_outs = \
            self.inference(all_samples, all_labels, batch_sizes, pair_batch_sizes, num_classes=num_classes, data_names=data_names, pair_flag=pair_flag)

        # calculate model loss
        # softmax loss (512 dim)
        for i, (weight_range_base, branch_logit, branch_label) in enumerate(zip(weight_range_bases, branch_logits, branch_labels)):
            self._arcface_sftmax_loss_distributed(branch_logit, branch_label, weight_range_base)

        if pair_flag:
            self._pair_loss(branch_feats=branch_outs[-4:], branch_labels=branch_labels[-4:], pair_left_feats=pair1_outs, pair_right_feats=pair2_outs,
                            add_black_feats=branch_outs[1:5], add_black_labels=branch_labels[1:5], scope='eccv')

        all_losses = tf.get_collection('loss_value', scope) + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(all_losses, name='total_loss')

        return total_loss