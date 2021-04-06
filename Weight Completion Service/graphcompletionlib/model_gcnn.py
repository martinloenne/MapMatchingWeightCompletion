from __future__ import division
import tensorflow as tf
import numpy as np
import scipy
from graphcompletionlib import graph
from graphcompletionlib.utils import show_all_variables

tfversion_ = tf.VERSION.split(".")
global tfversion
if int(tfversion_[0]) < 1:
    raise EnvironmentError("TF version should be above 1.0!!")

if int(tfversion_[1]) < 1:
    print("Working in TF version 1.0....")
    tfversion = "old"
else:
    print("Working in TF version 1.%d...." % int(tfversion_[1]))
    tfversion = "new"

EPSILON = 1e-3

class Model(object):
    """
    Defined:
        Placeholder
        Model architecture
        Train / Test function
    """
    
    def __init__(self, config, L, output_node, E=None):

        F = config.num_kernels
        K = config.conv_size
        p = config.pool_size
        M = config.FC_size
        M.append(output_node)

        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        # Powers of 2. Why? To fasten pooling, creating balanced binary tree
        assert np.all(np.mod(p_log2, 1) == 0)
        # When do pooling, we need these coarsenings to do operation.
        # Enough coarsening levels for pool sizes.
        assert len(L) >= 1 + np.sum(p_log2)

        if config.conv == 'gcnn':
            M_0 = L[0].shape[0]
        elif config.conv == 'cnn':
            M_0 = output_node
        else:
            raise ValueError(
                "Unsupported config.conv {}".format(
                    config.conv))
        # assign nodes after pooling
        j = 0
        self.L = []
        print("Pooling Size: ", p)
        for pp in p:
            self.L.append(L[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        print("The Node Information: ", self.L)

        self.F, self.K, self.p, self.M = F, K, p, M
        self.E = E
        self.filter = getattr(self, config.filter)
        self.brelu = getattr(self, config.brelu)
        self.pool = getattr(self, config.pool)
        self.regularizers = []

        self.regularization = config.regularization
        self.dropout = config.dropout
        self.model_type = config.model_type
        self.batch_size = config.batch_size
        self.num_node = M_0
        self.output_node = output_node
        self.feat_in = len(config.hist_range) - 1
        self.feat_out = len(config.hist_range) - 1

        self.classif_loss = config.classif_loss
        print("Config Learning Rate: ", config.learning_rate)
        self.start_lr = config.learning_rate
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate
        self.max_grad_norm = None
        if config.max_grad_norm > 0:
            self.max_grad_norm = config.max_grad_norm
        self.optimizer = config.optimizer
        
        self._build_placeholders()
        self._build_model()
        self._build_steps()
        self._build_optim()
        
        show_all_variables()

    def _build_placeholders(self):
        self.cnn_input = tf.placeholder(tf.float32,
                                        [self.batch_size, self.num_node, self.feat_in],
                                        name="cnn_input")
        self.output_label = tf.placeholder(tf.float32,
                                         [self.batch_size, self.output_node, self.feat_out],
                                         name="final_output")
        self.ph_labels_weight = tf.placeholder(
            tf.float32, (self.batch_size, self.output_node), 'labels_weight')

        # Place holder for embedding layer if any
        if self.E is not None:
            self.ph_embeds = []
            for i, E_i in enumerate(self.E):
                self.ph_embeds.append(tf.placeholder(
                    tf.int32, (self.batch_size, 1), name='embed_{}'.format(i)))
        else:
            self.ph_embeds = None

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.model_step = tf.Variable(0, name='model_step', trainable=False, dtype=tf.int32)
        print("Starting lr: ", self.start_lr)
        self.learning_rate = tf.train.exponential_decay(
            self.start_lr, self.model_step, self.decay_step,
            self.decay_rate, staircase=False)
        # self.learning_rate = self.start_lr

    def _build_model(self, reuse=None, bn=False):
        with tf.variable_scope("gconv_model", reuse=reuse) as sc:
            x = tf.expand_dims(self.cnn_input, 2)  # N x M x F x B
            for i, _ in enumerate(self.p):
                with tf.variable_scope('conv{}'.format(i + 1)):
                    with tf.name_scope('filter'):
                        if bn:
                            x = tf.layers.batch_normalization(
                                x, axis=1, training=self.is_training)
                        x = self.filter(x, self.L[i], self.F[i], self.K[i])
                    # with tf.name_scope('bucket_conv'):
                    #     x = self.bucket_conv(x)
                    with tf.name_scope('bias_relu'):
                        x = self.brelu(x)
                    with tf.name_scope('pooling'):
                        x = self.pool(x, self.p[i])
                    x = tf.layers.dropout(
                        x, rate=self.dropout, training=self.is_training)

            N, M, F, B = x.get_shape()
            print("The number of output node is ", M)

            list_tensor = []
            for k in range(B):
                hist_i = x[:, :, :, k]
                hist_i = tf.reshape(hist_i, (int(N), int(M * F)))  # N x M
                # concatenate with embedding layer
                embedding_x = []
                embedding_x.append(hist_i)
                if self.E is not None:
                    for i, E_i in enumerate(self.E):
                        with tf.device('/cpu:0'), tf.variable_scope('embed{}'.format(i)):
                            e_x = self.embed(self.ph_embeds[i], E_i, index=k)
                            ex_batch, e_x_timestep, e_x_embeded = e_x.get_shape()
                            e_x = tf.reshape(
                                e_x, [int(ex_batch * e_x_timestep), int(e_x_embeded)])
                            embedding_x.append(e_x)
                hist_i = tf.concat(embedding_x, 1)
                hist_i = tf.expand_dims(hist_i, -1)
                list_tensor.append(hist_i)
            x = tf.concat(list_tensor, axis=-1)

            # Fully connected hidden layers.
            for i, M in enumerate(self.M[:-1]):
                with tf.variable_scope('fc{}'.format(i + 1)):
                    if bn:
                        x = tf.layers.batch_normalization(
                            x, axis=1, training=self.is_training)
                    x = self.fc(x, M)
                    x = tf.layers.dropout(
                        x, rate=self.dropout, training=self.is_training)

            # fullly connected layer with normalization afterwards.
            with tf.variable_scope('logits'):
                x = self.fc(x, self.M[-1], relu=False)
                self.predictions = tf.nn.softmax(x, dim=-1)
                print("Info prediction is ", self.predictions)

            self.model_vars = tf.contrib.framework.get_variables(
                sc, collection=tf.GraphKeys.TRAINABLE_VARIABLES)
            
        self._build_loss()

    def _build_loss(self):
        if self.classif_loss == "kl":
            loss_batchmean = self.weighted_kl_tf(
                self.output_label, self.predictions, self.ph_labels_weight)
        elif self.classif_loss == 'l2':
            loss_batchmean = self.weighted_l2(
                self.output_label, self.predictions, self.ph_labels_weight)
        elif self.classif_loss == 'kl+l2':
            kl_loss = self.weighted_kl_tf(
                self.output_label, self.predictions, self.ph_labels_weight)
            l2_loss = self.weighted_l2(
                self.output_label, self.predictions, self.ph_labels_weight)
            loss_batchmean = kl_loss + 50 * l2_loss
        elif self.classif_loss == 'cdf_l2':
            loss_batchmean = self.weighted_l2_cdf(
                self.output_label, self.predictions, self.ph_labels_weight)
        elif self.classif_loss == 'cdf_l2+l2':
            cdf_l2 = self.weighted_l2_cdf(
                self.output_label, self.predictions, self.ph_labels_weight)
            l2_loss = self.weighted_l2(
                self.output_label, self.predictions, self.ph_labels_weight)
            loss_batchmean = cdf_l2 + l2_loss
        elif self.classif_loss == 'cdf_l2+kl':
            cdf_l2 = self.weighted_l2_cdf(
                self.output_label, self.predictions, self.ph_labels_weight)
            kl_loss = self.weighted_kl_tf(
                self.output_label, self.predictions, self.ph_labels_weight)
            loss_batchmean = cdf_l2 + 0.05 * kl_loss
        elif self.classif_loss == 'cross_entropy':
            loss_batchmean = self.weighted_cross_entropy(
                self.output_label, self.predictions, self.ph_labels_weight)
        elif self.classif_loss == 'max_abs_cdf':
            loss_batchmean = self.weighted_max_abs_diff(
                self.output_label, self.predictions, self.ph_labels_weight)


        with tf.name_scope("losses"):
            self.kl_loss = loss_batchmean

        if len(self.regularizers) > 0:
           with tf.name_scope('regularization'):
               regularization = self.regularization * tf.add_n(self.regularizers)
        else:
            regularization = 0

        self.loss = self.kl_loss + regularization

        self.model_summary = tf.summary.merge(
            [tf.summary.scalar("model_loss/loss",
                               self.kl_loss),
             tf.summary.scalar("model_loss/regularization",
                               regularization),
             tf.summary.scalar("model_loss/loss_reg",
                               self.loss)])
            
    def _build_steps(self):
        def run(sess, feed_dict, fetch,
                summary_op, summary_writer, output_op=None, output_img=None):
            if summary_writer is not None:
                fetch['summary'] = summary_op
            if output_op is not None:
                fetch['output'] = output_op

            result = sess.run(fetch, feed_dict=feed_dict)
            if "summary" in result.keys() and "step" in result.keys():
                summary_writer.add_summary(result['summary'], result['step'])
                summary_writer.flush()
            return result
        
        def train(sess, feed_dict, summary_writer=None,
                  with_output=False):
            fetch = {'loss': self.kl_loss,
                     'optim': self.model_optim, #?
                     'step': self.model_step, #?
                     'lr': self.learning_rate
            }
            return run(sess, feed_dict, fetch,
                       self.model_summary, summary_writer,
                       output_op=self.output_label if with_output else None,)
        
        def test(sess, feed_dict, summary_writer=None,
                 with_output=False):
            fetch = {'loss': self.kl_loss,
                     'pred': self.predictions,
                    'step': self.model_step}
            return run(sess, feed_dict, fetch,
                       self.model_summary, summary_writer,
                       output_op=self.output_label if with_output else None,)
        self.train = train
        self.test = test
        
    def _build_optim(self):
        def minimize(loss, step, var_list, learning_rate, optimizer):
            if optimizer == "sgd":
                optim = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == "adam":
                optim = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == "rmsprop":
                optim = tf.train.RMSPropOptimizer(learning_rate)
            else:
                raise Exception("[!] Unkown optimizer: {}".format(
                    optimizer))
            ## Gradient clipping ##    
            if self.max_grad_norm is not None:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None and var in var_list:
                        grad = tf.clip_by_norm(grad, self.max_grad_norm)
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}".format(
                                var.name))
                        new_grads_and_vars.append((grad, var))
                return optim.apply_gradients(new_grads_and_vars, global_step=step)
            else:
                grads_and_vars = optim.compute_gradients(
                    loss, var_list=var_list)
                return optim.apply_gradients(grads_and_vars,
                                             global_step=step)
        
        # optim #
        self.model_optim = minimize(
            self.loss,
            self.model_step,
            self.model_vars,
            self.learning_rate,
            self.optimizer)

    def embed(self, x, e_size, index=0):
        """"
        Embedding layer with input and embedded dimension

        x: the tf.placeholder of embedding layer
        e_size: the input and output size of current embedding layer
        """
        W = self._embedding_variable(e_size, index)
        embedded = tf.nn.embedding_lookup(W, x)

        return embedded

    def _weight_variable(self, shape, index=0, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable(
            'weights_{}'.format(index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def _embedding_variable(self, shape, index=0):
        var = tf.get_variable('embedding_{}'.format(index), shape, tf.float32)
        tf.summary.histogram(var.op.name, var)

        return var

    def _bias_variable(self, shape, index=0, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias_{}'.format(
            index), shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)

        return var

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min, B = x.get_shape()
        list_tensor = []
        for i in range(B):
            hist_i = x[:, :, i]

            W = self._weight_variable(
                [int(Min), Mout], index=i, regularization=True)
            b = self._bias_variable([Mout], index=i, regularization=True)
            hist_i = tf.matmul(hist_i, W) + b
            hist_i = tf.nn.relu(hist_i) if relu else hist_i
            hist_i = tf.expand_dims(hist_i, axis=-1)
            list_tensor.append(hist_i)
        fc_result = tf.concat(list_tensor, axis=-1)

        return fc_result

    # NN layers
    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F, B = x.get_shape()
        N, M, F, B = int(N), int(M), int(F), int(B)
        b = self._bias_variable([1, 1, F, B], regularization=True)

        return tf.nn.tanh(x + b)

    def mpool1(self, x, p):
        """
        Max pooling of size p.
        The size of the input x is [batch, len_feature, nb_kernels, nb_bins].

        x: [batch, height, width, channels]
        """

        if p > 1:
            x = tf.nn.max_pool(x, ksize=[1, p, 1, 1], strides=[
                1, p, 1, 1], padding='SAME')
            # tf.maximum
            return x  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """
        Average pooling of of size p.
        The size of the input x is [batch, len_feature, nb_kernels, nb_bins].

        """
        if p > 1:
            x = tf.nn.avg_pool(x, ksize=[1, p, 1, 1], strides=[
                1, p, 1, 1], padding='SAME')

            return x  # N x M/p x F x B
        else:
            return x  # N x M x F x B

    def chebyshev5(self, x, L, Fout, K):
        N, M, Fin, B = x.get_shape()
        N, M, Fin, B = int(N), int(M), int(Fin), int(B)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify
        # the shared L.

        L = scipy.sparse.csr_matrix(L)
        L = graph.rescale_L(L, lmax=2)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data.astype(np.float32), L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        list_tensor = []
        for i in range(B):
            hist_i = x[:, :, :, i]
            hist_i = tf.reshape(hist_i, (N, M, Fin))
            x0 = tf.transpose(hist_i, perm=[1, 2, 0])  # M x Fin x N
            x0 = tf.reshape(x0, [M, Fin * N])  # M x Fin*N
            hist_i = tf.expand_dims(x0, 0)  # 1 x M x Fin*N

            def concat(x, x_):
                x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
                return tf.concat([x, x_], axis=0)  # K x M x Fin*N

            # xk = 2 * L^{hat} * x_{k-1} - x_{k-2}, x_0 = x, x_1 = L * x
            if K > 1:
                x1 = tf.sparse_tensor_dense_matmul(L, x0)
                hist_i = concat(hist_i, x1)

            for k in range(2, K):
                x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
                hist_i = concat(hist_i, x2)
                x0, x1 = x1, x2

            hist_i = tf.reshape(hist_i, [K, M, Fin, N])  # K x M x Fin x N
            hist_i = tf.transpose(hist_i, perm=[3, 1, 2, 0])  # N x M x Fin x K
            hist_i = tf.reshape(hist_i, [N * M, Fin * K])  # N*M x Fin*K
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature
            # pair.
            W = self._weight_variable(
                [Fin * K, Fout], index=i, regularization=True)
            hist_i = tf.matmul(hist_i, W)  # N*M x Fout
            hist_i = tf.reshape(hist_i, [N, M, Fout])

            hist_i = tf.expand_dims(hist_i, -1)
            list_tensor.append(hist_i)
        gconvoluted = tf.concat(list_tensor, axis=-1)

        return gconvoluted  # N x M x Fout x B

    def conv2(self, x, L, Fout, K):
        N, M, Fin, B = x.get_shape()
        N, M, Fin, B = int(N), int(M), int(Fin), int(B)

        x = tf.transpose(x, perm=[0, 1, 3, 2])  # N x F x B x Channel
        # Do convolution on the axis of histogram, size=2
        f_w = int(2) if self.nb_bins > 2 else 1
        filter_size = [K, f_w, Fin, Fout]
        filter_kernel = self._weight_variable(filter_size, index=1, regularization=False)
        stride_size = [1, 1, 1, 1]
        x = tf.nn.conv2d(x, filter_kernel, stride_size, padding='SAME')

        return x

    def bucket_conv(self, x):
        N, M, Fin, B = x.get_shape()
        N, M, Fin, B = int(N), int(M), int(Fin), int(B)

        x = tf.transpose(x, perm=[0, 1, 3, 2])  # N x F x B x Channel
        # Do convolution on the axis of histogram, size=2
        # Input size: [batch, in_height, in_width, in_channels]
        # Filter Size: [filter_height, filter_width, in_channels, out_channels]
        f_w = int(2)
        filter_size = [1, f_w, Fin, Fin]
        filter_kernel = self._weight_variable(filter_size, index=1, regularization=False)
        stride_size = [1, 1, 1, 1]
        x = tf.nn.conv2d(x, filter_kernel, stride_size, padding='SAME')

        return x

    def conv1(self, x, L, Fout, K):
        N, M, Fin, B = x.get_shape()
        N, M, Fin, B = int(N), int(M), int(Fin), int(B)

        list_tensor = []
        for i in range(B):
            hist_i = x[:, :, :, i]

            # Shape info [batch, in_width, in_channels] of x
            # Shape info of filter [filter_width, in_channels, out_channels]
            # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature
            # pair.
            W = self._weight_variable(
                [K, Fin, Fout], index=i, regularization=False)
            hist_i = tf.nn.conv1d(hist_i, W, stride=1, padding='SAME')
            hist_i = tf.expand_dims(hist_i, -1)
            list_tensor.append(hist_i)
        convoluted = tf.concat(list_tensor, -1)

        return convoluted  # [batch, out_width, out_channels, nb_bins]

    def log10(self, x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def weighted_kl_tf(self, y_true, y_pred, weight, epsilon=EPSILON):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        log_pred = self.log10(y_pred + epsilon)
        log_true = self.log10(y_true + epsilon)
        log_sub = tf.subtract(log_pred, log_true)
        mul_op = tf.multiply(y_pred, log_sub)
        sum_hist = tf.reduce_sum(mul_op, 2)
        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div

    def weighted_kl_tf_true(self, y_true, y_pred, weight, epsilon=EPSILON):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        log_pred = tf.log(y_pred + epsilon)
        log_true = tf.log(y_true + epsilon)
        log_sub = tf.subtract(log_true, log_pred)
        mul_op = tf.multiply(y_true, log_sub)
        sum_hist = tf.reduce_sum(mul_op, 2)
        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div

    def weighted_cross_entropy(self, y_true, y_pred, weight, epsilon=1e-8):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        log_pred = tf.log(y_pred + epsilon)
        mul_op = tf.multiply(y_true, log_pred)
        sum_hist = tf.reduce_sum(mul_op, 2) * -1.
        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div

    def weighted_l2(self, y_true, y_pred, weight, epsilon=1e-8):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        y_sub = tf.subtract(y_pred, y_true)
        y_sub_2 = tf.square(y_sub)
        sum_hist = tf.reduce_sum(y_sub_2, 2)
        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div

    def weighted_l2_cdf(self, y_true, y_pred, weight):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        ones_matrix = tf.ones((B, B))
        mat_cdf_m = tf.matrix_band_part(ones_matrix, -1, 0)
        mat_cdf_m_2 = tf.matrix_band_part(ones_matrix, -1, 1)

        y_true = tf.transpose(y_true, [2, 1, 0]) # BxMxN
        y_pred = tf.transpose(y_pred, [2, 1, 0])
        y_true = tf.reshape(y_true, [B, M*N])
        y_pred = tf.reshape(y_pred, [B, M*N])
        cdf_true = tf.matmul(mat_cdf_m_2, y_true)
        cdf_pred = tf.matmul(mat_cdf_m, y_pred)

        cdf_true2 = tf.matmul(mat_cdf_m, y_true)
        cdf_pred2 = tf.matmul(mat_cdf_m_2, y_pred)

        cdf_true = tf.reshape(cdf_true, [B, M, N])
        cdf_pred = tf.reshape(cdf_pred, [B, M, N])
        cdf_true2 = tf.reshape(cdf_true2, [B, M, N])
        cdf_pred2 = tf.reshape(cdf_pred2, [B, M, N])

        cdf_true = tf.transpose(cdf_true, [2, 1, 0])
        cdf_pred = tf.transpose(cdf_pred, [2, 1, 0])
        cdf_true2 = tf.transpose(cdf_true2, [2, 1, 0])
        cdf_pred2 = tf.transpose(cdf_pred2, [2, 1, 0])

        y_sub = tf.subtract(cdf_true, cdf_pred)
        y_sub_2 = tf.abs(y_sub)

        y_sub2 = tf.subtract(cdf_true2, cdf_pred)
        y_sub2_2 = tf.abs(y_sub2)

        y_sub3 = tf.subtract(cdf_true, cdf_pred2)
        y_sub3_2 = tf.abs(y_sub3)

        max_hist = tf.maximum(y_sub_2, y_sub2_2)
        max_hist = tf.maximum(max_hist, y_sub3_2)
        sum_hist = tf.reduce_max(max_hist, 2)

        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
            #        avg_kl_div = tf.reduce_mean(sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div

    def weighted_max_abs_diff(self, y_true, y_pred, weight):

        N, M, B = y_pred.get_shape()
        N, M, B = int(N), int(M), int(B)
        w_N, w_M = weight.get_shape()
        w_N, w_M = int(w_N), int(w_M)
        assert w_N == N, w_M == M

        ones_matrix = tf.ones((B, B))
        mat_cdf_m = tf.matrix_band_part(ones_matrix, -1, 0)

        y_true = tf.transpose(y_true, [2, 1, 0])  # BxMxN
        y_pred = tf.transpose(y_pred, [2, 1, 0])
        y_true = tf.reshape(y_true, [B, M * N])
        y_pred = tf.reshape(y_pred, [B, M * N])
        cdf_pred = tf.matmul(mat_cdf_m, y_pred)
        cdf_true = tf.matmul(mat_cdf_m, y_true)

        cdf_true = tf.reshape(cdf_true, [B, M, N])
        cdf_pred = tf.reshape(cdf_pred, [B, M, N])
        cdf_true = tf.transpose(cdf_true, [2, 1, 0])
        cdf_pred = tf.transpose(cdf_pred, [2, 1, 0])
        y_sub = tf.subtract(cdf_true, cdf_pred)
        y_sub_2 = tf.abs(y_sub)
        sum_hist = tf.reduce_max(y_sub_2, 2)

        if weight is not None:
            sum_hist = tf.multiply(weight, sum_hist)
        weight_avg_kl_div = tf.reduce_sum(sum_hist)
        avg_kl_div = weight_avg_kl_div / tf.reduce_sum(weight)

        return avg_kl_div
