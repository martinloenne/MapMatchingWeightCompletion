import numpy as np
import tensorflow as tf
from tqdm import trange
from graphcompletionlib import graph
from graphcompletionlib.model_gcnn import Model
from graphcompletionlib.utils import BatchLoader

"""
Trainer: 

1. Initializes model
2. Train
3. Test
"""


class Trainer(object):
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.model_dir = config.model_dir
        self.gpu_memory_fraction = config.gpu_memory_fraction
        self.checkpoint_secs = config.checkpoint_secs
        self.log_step = config.log_step
        self.num_epoch = config.num_epochs
        self.stop_win_size = config.stop_win_size
        self.stop_early = config.stop_early

        ## import data Loader ##ir
        batch_size = config.batch_size
        server_name = config.server_name
        mode = config.mode
        target = config.target
        sample_rate = config.sample_rate
        win_size = config.win_size
        hist_range = config.hist_range
        s_month = config.s_month
        e_month = config.e_month
        e_date = config.e_date
        s_date = config.s_date
        data_rm = config.data_rm
        coarsening_level = config.coarsening_level
        cnn_mode = config.conv
        is_coarsen = config.is_coarsen
        is_predicting = config.is_predicting


        self.data_loader = BatchLoader(server_name, mode, target, sample_rate, win_size,
                                       hist_range, s_month, s_date, e_month, e_date,
                                       data_rm, batch_size, coarsening_level, cnn_mode,
                                       is_coarsen, is_predicting)

        actual_node = self.data_loader.adj.shape[0]
        if config.conv == 'gcnn':
            graphs = self.data_loader.graphs
            if config.is_coarsen:
                L = [graph.laplacian(A, normalized=config.normalized) for A in graphs]
            else:
                L = [graph.laplacian(self.data_loader.adj,
                                      normalized=config.normalized)] * len(graphs)
        elif config.conv == 'cnn':
            L = [actual_node]
            tmp_node = actual_node
            while tmp_node > 0:
                tmp_node = int(tmp_node / 2)
                L.append(tmp_node)
        else:
            raise ValueError(
                "Unsupported config.conv {}".format(
                    config.conv))

        tf.reset_default_graph()
        ## define model ##
        self.model = Model(config, L, actual_node)

        ## model saver / summary writer ##
        self.saver = tf.train.Saver()
        self.model_saver = tf.train.Saver(self.model.model_vars)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        # Checkpoint
        # meta file: describes the saved graph structure, includes
        # GraphDef, SaverDef, and so on; then apply
        # tf.train.import_meta_graph('/tmp/model.ckpt.meta'),
        # will restore Saver and Graph.

        # index file: it is a string-string immutable
        # table(tensorflow::table::Table). Each key is a name of a tensor
        # and its value is a serialized BundleEntryProto.
        # Each BundleEntryProto describes the metadata of a
        # tensor: which of the "data" files contains the content of a tensor,
        # the offset into that file, checksum, some auxiliary data, etc.
        #
        # data file: it is TensorBundle collection, save the values of all variables.
        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_summaries_secs=300,
                                 save_model_secs=self.checkpoint_secs,
                                 global_step=self.model.model_step)

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=self.gpu_memory_fraction,
            allow_growth=True)  # seems to be not working
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        #
        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        # init = tf.global_variables_initializer()
        # self.sess = tf.Session(config=sess_config)
        # self.sess.run(init)

    def train(self, val_best_score=10, save=False, index=1, best_model=None):
        print("[*] Checking if previous run exists in {}"
              "".format(self.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if tf.train.latest_checkpoint(self.model_dir) is not None:
            print("[*] Saved result exists! loading...")
            self.saver.restore(
                self.sess,
                latest_checkpoint
            )
            print("[*] Loaded previously trained weights")
            self.b_pretrain_loaded = True
        else:
            print("[*] No previous result")
            self.b_pretrain_loaded = False

        print("[*] Training starts...")
        self.model_summary_writer = None

        val_loss = 0
        lr = 0
        tmp_best_loss = float('+inf')
        validation_loss_window = np.zeros(self.stop_win_size)
        validation_loss_window[:] = float('+inf')
        ##Training
        for n_epoch in trange(self.num_epoch, desc="Training[epoch]"):
            self.data_loader.reset_batch_pointer(0)
            loss_epoch = []
            for k in trange(self.data_loader.sizes[0], desc="[per_batch]"):
                # Fetch training data
                batch_x, batch_y, weight_y,\
                count_y, _ = self.data_loader.next_batch(0)

                feed_dict = {
                    self.model.cnn_input: batch_x,
                    self.model.output_label: batch_y,
                    self.model.ph_labels_weight: weight_y,
                    self.model.is_training: True
                }
                res = self.model.train(self.sess, feed_dict, self.model_summary_writer,
                                       with_output=True)
                loss_epoch.append(res['loss'])
                lr = res['lr']
                self.model_summary_writer = self._get_summary_writer(res)

            val_loss = self.validate()
            train_loss = np.mean(loss_epoch)

            validation_loss_window[n_epoch % self.stop_win_size] = val_loss

            if self.stop_early:
                if np.abs(validation_loss_window.mean() - val_loss) < 1e-4:
                    print('Validation loss did not decrease. Stopping early.')
                    break

            if n_epoch % 10 == 0:
                if save:
                    self.saver.save(self.sess, self.model_dir)
                if val_loss < val_best_score:
                    val_best_score = val_loss
                    best_model = self.model_dir
                if val_loss < tmp_best_loss:
                    tmp_best_loss = val_loss
                print("Searching {}...".format(index))
                print("Epoch {}: ".format(n_epoch))
                print("LR: ", lr)
                print("  Train Loss: ", train_loss)
                print("  Validate Loss: ", val_loss)
                print("  Current Best Loss: ", val_best_score)
                print("  Current Model Dir: ", best_model)

        return tmp_best_loss

    def validate(self):

        loss = []
        for n_sample in trange(self.data_loader.sizes[1], desc="Validating"):
            batch_x, batch_y, weight_y, count_y,\
            _ = self.data_loader.next_batch(1)

            feed_dict = {
                self.model.cnn_input: batch_x,
                self.model.output_label: batch_y,
                self.model.ph_labels_weight: weight_y,
                self.model.is_training: False
            }
            res = self.model.test(self.sess, feed_dict, self.summary_writer,
                                  with_output=True)
            loss.append(res['loss'])

        return np.nanmean(loss)

    def test(self):

        loss = []
        gt_y = []
        pred_y = []
        w_y = []
        counts_y = []
        vel_list_y = []
        for n_sample in trange(self.data_loader.sizes[2], desc="Testing"):
            batch_x, batch_y, weight_y, \
            count_y, vel_list = self.data_loader.next_batch(2)

            feed_dict = {
                self.model.cnn_input: batch_x,
                self.model.output_label: batch_y,
                self.model.ph_labels_weight: weight_y,
                self.model.is_training: False
            }
            res = self.model.test(self.sess, feed_dict, self.summary_writer,
                                  with_output=True)
            loss.append(res['loss'])
            gt_y.append(batch_y)
            w_y.append(weight_y)
            counts_y.append(count_y)
            vel_list_y.append(vel_list)
            pred_y.append(res['pred'])

        final_gt = np.concatenate(gt_y, axis=0)
        final_pred = np.concatenate(pred_y, axis=0)
        final_weight = np.concatenate(w_y, axis=0)
        final_count = np.concatenate(counts_y, axis=0)
        final_vel_list = np.concatenate(vel_list_y, axis=0)

        result_dict = {'ground_truth': final_gt,
                       'prediction': final_pred,
                       'weight': final_weight,
                       'count': final_count,
                       'vel_list': final_vel_list}

        test_loss = np.mean(loss)
        print("Test Loss: ", test_loss)

        return result_dict
            # self.model_summary_writer = self._get_summary_writer(res)

    def _get_summary_writer(self, result):
        if result['step'] % self.log_step == 0:
            return self.summary_writer
        else:
            return None

