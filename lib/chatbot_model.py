from __future__ import print_function

import copy as copy
import datetime
import filecmp
import hashlib
import json
import os
import os.path
import re
import shutil
import platform

import MeCab
# noinspection PyUnresolvedReferences
import easy_tf_log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
# noinspection PyUnresolvedReferences
from easy_tf_log import tflog
# noinspection PyUnresolvedReferences
from google.colab import files
# noinspection PyUnresolvedReferences
from pushbullet import Pushbullet
from tensorflow.python.layers import core as layers_core
from tensorflow.python.platform import gfile
from enum import Enum, auto


class Mode(Enum):
    Test = auto()
    TrainSeq2Seq = auto()
    TrainSeq2SeqSwapped = auto()
    TrainRL = auto()
    TweetBot = auto()


def pp(*arguments):
    print(*arguments)
    with open("stdout.txt", "a") as fout:
        print(*arguments, file=fout)


def is_local():
    return platform.system() == 'Darwin'


def client_id():
    if is_local():
        return "local"
    # noinspection SpellCheckingInspection
    clients = {'dfc1d5b22ba03430800179d23e522f6f': 'client1',
               'f8e857a2d792038820ebb2ae8d803f7c': 'client2',
               '7628f983785173edabbde501ef8f781d': 'client3'}
    with open('/content/datalab/adc.json') as json_data:
        d = json.load(json_data)
        email = d['id_token']['email'].encode('utf-8')
        return clients[hashlib.md5(email).hexdigest()]


if is_local():
    drive_path = '/Users/higepon/Google Drive/seq2seq_data'
else:
    drive_path = 'drive/seq2seq_data'

pp(client_id())
current_client_id = client_id()

mode = Mode.Test


# mode = Mode.TrainSeq2Seq
# mode = Mode.TrainSeq2SeqSwapped
# mode = Mode.TrainRL
# mode = Mode.TweetBot

class DeltaLogger:
    def __init__(self, key, step, stdout=None):
        self.key = key
        self.step = step
        self.stdout = stdout

    def __enter__(self):
        self.start_time = datetime.datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = datetime.datetime.now()
        delta_sec = (end_time - self.start_time).total_seconds()

        tflog("{}[{}]".format(self.key, current_client_id), delta_sec,
              step=self.step)
        if self.stdout is not None:
            pp("1{}={}".format(self.key, round(delta_sec, 1)))
        if exc_type is None:
            return False


def delta(key, step, stdout=False):
    return DeltaLogger(key, step, stdout)


class Shell:
    @staticmethod
    def download_file_if_necessary(file_name):
        if os.path.exists(file_name):
            return
        pp("downloading {}...".format(file_name))
        shutil.copy2(os.path.join(drive_path, file_name), file_name)
        pp("downloaded")

    @staticmethod
    def download_model_data_if_necessary(model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        pp("Downloading model files...")
        src_dir = os.path.join(drive_path, model_path)
        Shell.copy_all_files(src_dir, model_path)
        pp("done")

    @staticmethod
    def copy_all_files(src_dir, dst_dir):
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                src = os.path.join(src_dir, file)
                dst = os.path.join(dst_dir, file)
                if os.path.exists(dst) and filecmp.cmp(src, dst):
                    pp("Skip copying ", src)
                    continue
                else:
                    pp("Copying ", src)
                shutil.copy2(src, dst)

    @staticmethod
    def remove_all_files(target_dir):
        for file in Shell.listdir(target_dir):
            os.remove(file)

    @staticmethod
    def remove_matched_files(target_dir, pattern):
        for file in Shell.listdir(target_dir):
            if re.match(pattern, file):
                os.remove(file)

    @staticmethod
    def download_logs(path):
        for file in Shell.listdir(path):
            if re.match('.*events', file):
                files.download(file)

    @staticmethod
    def download(path):
        files.download(path)

    @staticmethod
    def remove_saved_model(hparams):
        os.makedirs(hparams.model_path, exist_ok=True)
        Shell.remove_all_files(hparams.model_path)
        os.makedirs(os.path.join(drive_path, hparams.model_path), exist_ok=True)
        Shell.remove_all_files(os.path.join(drive_path, hparams.model_path))

    @staticmethod
    def copy_saved_model(src_hparams, dst_hparams):
        Shell.copy_all_files(src_hparams.model_path, dst_hparams.model_path)
        # rm tf.logs from source so that it wouldn't be mixed in dest tf.logs.
        Shell.remove_matched_files(dst_hparams.model_path, ".*events.*")

    @staticmethod
    def listdir(target_dir):
        for dir_path, _, file_names in os.walk(target_dir):
            for file in file_names:
                yield os.path.abspath(os.path.join(dir_path, file))

    @staticmethod
    def list_model_file(path):
        file = open('{}/checkpoint'.format(path))
        text = file.read()
        file.close()
        pp("model_file", text)
        m = re.match(r".*ChatbotModel-(\d+)", text)
        model_name = m.group(1)
        files = ["checkpoint"]
        files.extend([x for x in os.listdir(path) if
                      re.search(model_name, x) or re.search('events.out', x)])
        return files

    @staticmethod
    def save_model_in_drive(model_path):
        path = os.path.join(drive_path, model_path)
        os.makedirs(path, exist_ok=True)
        Shell.remove_all_files(os.path.join(drive_path, model_path))
        pp("Saving model in Google Drive...")
        for file in Shell.list_model_file(model_path):
            pp("Saving ", file)
            shutil.copy2(os.path.join(model_path, file),
                         os.path.join(drive_path, model_path, file))
        pp("done")


config_path = 'config.yml'
Shell.download_file_if_necessary(config_path)
f = open(config_path, 'rt')
push_key = yaml.load(f)['pushbullet']['api_key']

pb = Pushbullet(push_key)

# Note for myself.
# You've summarized Seq2Seq
# at http://d.hatena.ne.jp/higepon/20171210/1512887715.

# If you see following error, it means your max(len(tweets of training set))
# <  decoder_length.
# This should be a bug somewhere in build_decoder, but couldn't find one yet.
# You can workaround by setting hparams.decoder_length=max len of tweet in
# training set.
# InvalidArgumentError: logits and labels must have the same first dimension,
#  got logits shape [48,50] and labels shape [54]
# [[Node: root/SparseSoftmaxCrossEntropyWithLogits
# /SparseSoftmaxCrossEntropyWithLogits = SparseSoftmaxCrossEntropyWithLogits[
# T=DT_FLOAT, Tlabels=DT_INT32,

pp(tf.__version__)


def has_gpu0():
    return tf.test.gpu_device_name() == "/device:GPU:0"


class ModelDirectory(Enum):
    tweet_large = 'model/tweet_large'
    tweet_large_rl = 'model/tweet_large_rl'
    tweet_large_swapped = 'model/tweet_large_swapped'
    tweet_small = 'model/tweet_small'
    tweet_small_swapped = 'model/tweet_small_swapped'
    tweet_small_rl = 'model/tweet_small_rl'
    conversations_small = 'model/conversations_small'
    conversations_small_backward = 'model/conversations_small_backward'
    conversations_small_rl = 'model/conversations_small_rl'
    conversations_large = 'model/conversations_large'
    conversations_large_backward = 'model/conversations_large_backward'
    conversations_large_rl = 'model/conversations_large_rl'
    test_multiple1 = 'model/test_multiple1'
    test_multiple2 = 'model/test_multiple2'
    test_multiple3 = 'model/test_multiple3'
    test_distributed = 'model/test_distributed'

    @staticmethod
    def create_all_directories():
        for d in ModelDirectory:
            os.makedirs(d.value, exist_ok=True)


ModelDirectory.create_all_directories()

base_hparams = tf.contrib.training.HParams(
    machine=current_client_id,
    batch_size=3,
    num_units=6,
    num_layers=2,
    vocab_size=9,
    embedding_size=8,
    learning_rate=0.01,
    learning_rate_decay=0.99,
    use_attention=False,
    encoder_length=5,
    decoder_length=5,
    max_gradient_norm=5.0,
    beam_width=2,
    num_train_steps=100,
    debug_verbose=False,
    model_path='Please override model_directory',
    sos_id=0,
    eos_id=1,
    pad_id=2,
    unk_id=3,
    sos_token="[SOS]",
    eos_token="[EOS]",
    pad_token="[PAD]",
    unk_token="[UNK]",
)

# For debug purpose.
tf.reset_default_graph()


class ChatbotModel:
    def __init__(self, sess, hparams, model_path, scope='ChatbotModel'):
        self.sess = sess
        # todo remove
        self.hparams = hparams

        # todo
        self.model_path = model_path
        self.scope = scope
        # Sampled replies in previous session,
        # this is necessary to back propagation.
        self.sampled = tf.placeholder(tf.int32, name="sampled")

        # Used to store previously inferred by beam_search.
        #        self.beam_predicted_ids = tf.placeholder(tf.int32,
        #
        # name="beam_predicted_ids")
        self.enc_inputs, self.enc_inputs_lengths, enc_outputs, enc_state, \
        emb_encoder = self._build_encoder(
            hparams, scope)

        self.dec_inputs, self.dec_tgt_lengths, self._logits, \
        self.sample_logits, self.sample_replies, \
        self.log_probs_selected, self.infer_logits, self.replies, \
        self.beam_replies = self._build_decoder(
            hparams, self.enc_inputs_lengths, emb_encoder,
            enc_state, enc_outputs)

        self._probs = tf.nn.softmax(self.infer_logits)
        self._log_probs = tf.nn.log_softmax(self.infer_logits)

        self.reward = tf.placeholder(tf.float32, name="reward")
        self.tgt_labels, self.global_step, self.loss, self.train_op = \
            self._build_seq2seq_optimizer(
                hparams, self._logits)
        self.rl_loss, self.rl_train_op = self._build_rl_optimizer(hparams)

        self.train_loss_summary = tf.summary.scalar("loss", self.loss)
        self.val_loss_summary = tf.summary.scalar("validation_loss",
                                                  self.loss)
        self.merged_summary = tf.summary.merge_all()

        # Initialize saver after model created
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    def train(self, enc_inputs, enc_inputs_lengths, target_labels,
              dec_inputs, dec_target_lengths):

        feed_dict = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
            self.tgt_labels: target_labels,
            self.dec_inputs: dec_inputs,
            self.dec_tgt_lengths: dec_target_lengths,
        }
        _, global_step, summary = self.sess.run(
            [self.train_op, self.global_step, self.train_loss_summary],
            feed_dict=feed_dict)

        return global_step, summary

    def infer(self, enc_inputs, enc_inputs_lengths):
        infer_feed_dic = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
        }
        return self.sess.run([self.replies, self.infer_logits],
                             feed_dict=infer_feed_dic)

    def log_probs(self, enc_inputs, enc_inputs_lengths):
        infer_feed_dic = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
        }
        return self.sess.run([self._log_probs, self._probs],
                             feed_dict=infer_feed_dic)

    def log_probs_sampled(self, enc_inputs, enc_inputs_lengths, sampled):
        infer_feed_dic = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
            self.sampled: sampled
        }
        return self.sess.run(
            [self.log_probs_selected, self.sample_logits],
            feed_dict=infer_feed_dic)

    def infer_beam_search(self, enc_inputs, enc_inputs_lengths):
        """
        :return: (replies: [batch_size, decoder_length, beam_size],
                  logits: [batch_size, decoder_length, vocab_size]))
        """
        infer_feed_dic = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
        }
        return self.sess.run([self.beam_replies, self.infer_logits, self._probs,
                              self._log_probs],
                             feed_dict=infer_feed_dic)

    def sample(self, enc_inputs, enc_inputs_lengths):
        infer_feed_dic = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
        }

        replies, logits = self.sess.run(
            [self.sample_replies, self.sample_logits],
            feed_dict=infer_feed_dic)
        return replies, logits

    def batch_loss(self, enc_inputs, enc_inputs_lengths, tgt_labels,
                   dec_inputs, dec_tgt_lengths):
        feed_dict = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
            self.tgt_labels: tgt_labels,
            self.dec_inputs: dec_inputs,
            self.dec_tgt_lengths: dec_tgt_lengths,
        }
        return self.sess.run([self.loss, self.val_loss_summary],
                             feed_dict=feed_dict)

    def seq_len(self, seq):
        try:
            # length includes the first eos_id.
            return seq.index(self.hparams.eos_id) + 1
        except ValueError:
            return self.hparams.encoder_length

    def train_with_reward(self, enc_inputs, enc_inputs_lengths, sampled,
                          reward):
        feed_dict = {
            self.enc_inputs: enc_inputs,
            self.enc_inputs_lengths: enc_inputs_lengths,
            self.sampled: sampled,
            self.reward: reward
        }

        _, global_step, loss = self.sess.run(
            [self.rl_train_op, self.global_step, self.rl_loss],
            feed_dict=feed_dict)
        return global_step, loss

    def save(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model_dir = "{}/{}".format(model_path, self.scope)
        self.saver.save(self.sess, model_dir, global_step=self.global_step)

    def restore(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, last_model)
            return True
        else:
            pp("Created fresh model.")
            return False

    @staticmethod
    def _softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _build_rl_optimizer(self, hparams):
        # todo mask the sampling results
        sample_log_prob_shape = tf.shape(self.log_probs_selected)
        reward_shape = tf.shape(self.reward)
        reward_shape_print = tf.Print(reward_shape,
                                      [reward_shape],
                                      message="reward_shape")
        reward_print = tf.Print(self.reward,
                                [self.reward],
                                message="reward")

        asserts = [tf.assert_equal(sample_log_prob_shape[0],
                                   reward_shape_print[0],
                                   [self.log_probs_selected,
                                    self.reward]),
                   tf.assert_equal(sample_log_prob_shape[1],
                                   reward_shape_print[1],
                                   [self.log_probs_selected,
                                    self.reward]), reward_print
                   ]
        with tf.control_dependencies(asserts):
            loss = -tf.reduce_sum(
                self.log_probs_selected * self.reward) / tf.to_float(
                hparams.batch_size)
        train_op = self._build_optimizer_with_loss(self.global_step, hparams,
                                                   loss)
        return loss, train_op

    def _build_optimizer_with_loss(self, global_step, hparams, loss):
        params = tf.trainable_variables()
        optimizer = tf.train.GradientDescentOptimizer(hparams.learning_rate)
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, hparams.max_gradient_norm)
        with tf.device(self.available_device()):
            train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=global_step)
        return train_op

    def _build_seq2seq_optimizer(self, hparams, logits):
        # Target labels
        #   As described in doc for sparse_softmax_cross_entropy_with_logits,
        #   labels should be [batch_size, decoder_target_lengths]
        #   instead of [batch_size, decoder_target_lengths, vocab_size].
        #   So labels should have indices instead of vocab_size classes.
        tgt_labels = tf.placeholder(tf.int32, shape=(
            hparams.batch_size, hparams.decoder_length), name="tgt_labels")
        # Loss
        #   tgt_labels: [batch_size, decoder_length]
        #   _logits: [batch_size, decoder_length, vocab_size]
        #   crossent: [batch_size, decoder_length]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tgt_labels, logits=logits)
        tgt_weights = tf.sequence_mask(self.dec_tgt_lengths,
                                       hparams.decoder_length,
                                       dtype=logits.dtype)
        crossent = crossent * tgt_weights
        crossent_by_batch = tf.reduce_sum(crossent, axis=1)
        loss = tf.reduce_sum(crossent_by_batch) / tf.to_float(
            hparams.batch_size)
        # Train
        global_step = tf.get_variable(name="global_step", shape=[],
                                      dtype=tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        train_op = self._build_optimizer_with_loss(global_step, hparams, loss)
        return tgt_labels, global_step, loss, train_op

    @staticmethod
    def available_device():
        device = '/cpu:0'
        if has_gpu0():
            device = '/gpu:0'
            pp("$$$ GPU ENABLED $$$")
        return device

    @staticmethod
    def _build_encoder(hparams, scope):
        # Encoder
        #   enc_inputs: [encoder_length, batch_size]
        #   This is time major where encoder_length comes
        #   first instead of batch_size.
        #   enc_inputs_lengths: [batch_size]
        enc_inputs = tf.placeholder(tf.int32, shape=(
            hparams.encoder_length, hparams.batch_size), name="enc_inputs")
        enc_inputs_lengths = tf.placeholder(tf.int32,
                                            shape=hparams.batch_size,
                                            name="enc_inputs_lengths")

        # Embedding
        #   We originally didn't share embedding between encoder and decoder.
        #   But now we share it. It makes much easier to calculate rewards.
        #   Matrix for embedding: [vocab_size, embedding_size]
        #   Should be shared between training and inference.
        with tf.variable_scope(scope):
            emb_encoder = tf.get_variable("emb_encoder",
                                          [hparams.vocab_size,
                                           hparams.embedding_size])

        # Look up embedding:
        #   enc_inputs: [encoder_length, batch_size]
        #   enc_emb_inputs: [encoder_length, batch_size, embedding_size]
        enc_emb_inputs = tf.nn.embedding_lookup(emb_encoder, enc_inputs)

        # LSTM cell.
        with tf.variable_scope(scope):
            # Should be shared between training and inference.
            cells = []
            for _ in range(hparams.num_layers):
                cells.append(
                    tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units))
            encoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

            # Run Dynamic RNN
            #   enc_outputs: [encoder_length, batch_size, num_units]
            #   enc_state: [batch_size, num_units],
            #   this is final state of the cell for each batch.
            enc_outputs, enc_state = tf.nn.dynamic_rnn(encoder_cell,
                                                       enc_emb_inputs,
                                                       time_major=True,
                                                       dtype=tf.float32,
                                                       sequence_length=enc_inputs_lengths)

        return enc_inputs, enc_inputs_lengths, enc_outputs, enc_state, \
               emb_encoder

    @staticmethod
    def _build_training_decoder(hparams, enc_inputs_lengths,
                                enc_state, enc_outputs, dec_cell,
                                dec_emb_inputs, dec_tgt_lengths,
                                projection_layer, scope):

        dynamic_batch_size = tf.shape(enc_inputs_lengths)[0]
        initial_state, wrapped_dec_cell = ChatbotModel._attention_wrapper(
            dec_cell, dynamic_batch_size, enc_inputs_lengths, enc_outputs,
            enc_state, hparams, scope, reuse=False)

        # Decoder with helper:
        #   dec_emb_inputs: [decoder_length, batch_size, embedding_size]
        #   dec_tgt_lengths: [batch_size] vector,
        #   which represents each target sequence length.
        with tf.variable_scope(scope):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                dec_emb_inputs,
                dec_tgt_lengths,
                time_major=True)

        # Decoder and decode
        with tf.variable_scope(scope):
            with tf.variable_scope("training"):
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    wrapped_dec_cell, training_helper, initial_state,
                    output_layer=projection_layer)

        # Dynamic decoding
        #   final_outputs.rnn_output: [batch_size, decoder_length,
        #                             vocab_size], list of RNN state.
        #   final_outputs.sample_id: [batch_size, decoder_length],
        #                            list of argmax of rnn_output.
        #   final_state: [batch_size, num_units],
        #                list of final state of RNN on decode process.
        #   final_sequence_lengths: [batch_size], list of each decoded sequence.
        with tf.variable_scope(scope):
            final_outputs, _final_state, _final_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(
                    training_decoder)

        if hparams.debug_verbose:
            pp("rnn_output.shape=", final_outputs.rnn_output.shape)
            pp("sample_id.shape=", final_outputs.sample_id.shape)
            pp("final_state=", _final_state)
            pp("final_sequence_lengths.shape=",
               _final_sequence_lengths.shape)

        logits = final_outputs.rnn_output
        return logits, wrapped_dec_cell, initial_state

    def _build_decoder(self, hparams, enc_inputs_lengths, embedding_encoder,
                       enc_state, enc_outputs):
        # Decoder input
        #   dec_inputs: [decoder_length, batch_size]
        #   dec_tgt_lengths: [batch_size]
        #   This is grand truth target inputs for training.
        dec_inputs = tf.placeholder(tf.int32, shape=(
            hparams.decoder_length, hparams.batch_size), name="dec_inputs")
        dec_tgt_lengths = tf.placeholder(tf.int32,
                                         shape=hparams.batch_size,
                                         name="dec_tgt_lengths")

        # Look up embedding:
        #   dec_inputs: [decoder_length, batch_size]
        #   decoder_emb_inp: [decoder_length, batch_size, embedding_size]
        dec_emb_inputs = tf.nn.embedding_lookup(embedding_encoder,
                                                dec_inputs)

        # https://stackoverflow.com/questions/39573188/output-projection-in
        # -seq2seq-model-tensorflow
        # Internally, a neural network operates on dense vectors of some size,
        # often 256, 512 or 1024 floats (let's say 512 for here).
        # But at the end it needs to predict a word
        # from the vocabulary which is often much larger,
        # e.g., 40000 words. Output projection is the final linear layer
        # that converts (projects) from the internal representation
        #  to the larger one.
        # So, for example, it can consist of a 512 x 40000 parameter matrix
        # and a 40000 parameter for the bias vector.
        projection_layer = layers_core.Dense(hparams.vocab_size, use_bias=False)

        # We share this between training and inference.
        cells = []
        for _ in range(hparams.num_layers):
            cells.append(tf.nn.rnn_cell.BasicLSTMCell(hparams.num_units))
        dec_cell = tf.contrib.rnn.MultiRNNCell(cells)

        # Training graph
        logits, wrapped_dec_cell, initial_state = self._build_training_decoder(
            hparams, enc_inputs_lengths, enc_state, enc_outputs,
            dec_cell, dec_emb_inputs, dec_tgt_lengths,
            projection_layer, self.scope)

        infer_logits, replies = self._build_greedy_inference(hparams,
                                                             embedding_encoder,
                                                             enc_state,
                                                             enc_inputs_lengths,
                                                             enc_outputs,
                                                             dec_cell,
                                                             projection_layer,
                                                             self.scope)

        # Beam Search Inference graph
        beam_replies = self._build_beam_search_inference(hparams,
                                                         enc_inputs_lengths,
                                                         embedding_encoder,
                                                         enc_state,
                                                         enc_outputs,
                                                         dec_cell,
                                                         projection_layer,
                                                         self.scope)

        # beam_log_probs = self._log_probs_beam(infer_logits,
        #                                       self.beam_predicted_ids)

        # Sample Inference graph
        _, sample_replies = self._build_sample_inference(hparams,
                                                         embedding_encoder,
                                                         enc_state,
                                                         enc_inputs_lengths,
                                                         enc_outputs,
                                                         dec_cell,
                                                         projection_layer,
                                                         self.scope)

        # Here we use infer_logits which is generated from argmax.
        # We don't use sample_logits for RL, because infer_logts and
        # sample_logits are different.
        # And eventually infer_logits should become our desiered inference
        # with our RL training.
        logits_print = tf.Print(infer_logits, [infer_logits],
                                message="infer_logits")
        indices = self._convert_indices(self.sampled)
        log_probs = tf.nn.log_softmax(logits_print)
        log_probs_selected = tf.gather_nd(log_probs, indices)
        return dec_inputs, dec_tgt_lengths, logits, logits_print, \
               sample_replies, log_probs_selected, infer_logits, \
               replies, beam_replies

    @staticmethod
    def _build_greedy_inference(hparams, embedding_encoder, enc_state,
                                encoder_inputs_lengths, encoder_outputs,
                                dec_cell, projection_layer, scope):
        # Greedy decoder
        dynamic_batch_size = tf.shape(encoder_inputs_lengths)[0]
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding_encoder,
            tf.fill([dynamic_batch_size], hparams.sos_id), hparams.eos_id)

        infer_logits, replies = ChatbotModel._dynamic_decode(dec_cell,
                                                             dynamic_batch_size,
                                                             encoder_inputs_lengths,
                                                             encoder_outputs,
                                                             enc_state,
                                                             hparams,
                                                             inference_helper,
                                                             projection_layer,
                                                             scope)
        return infer_logits, replies

    @staticmethod
    def _build_beam_search_inference(hparams, encoder_inputs_lengths,
                                     embedding_encoder, enc_state,
                                     encoder_outputs, dec_cell,
                                     projection_layer, scope):

        assert (hparams.beam_width != 0)

        dynamic_batch_size = tf.shape(encoder_inputs_lengths)[0]
        # https://github.com/tensorflow/tensorflow/issues/11904
        if hparams.use_attention:
            with tf.variable_scope(scope, reuse=True):
                # Attention
                # encoder_outputs is time major, so transopse it to batch major.
                # attention_encoder_outputs: [batch_size, encoder_length,
                # num_units]
                attention_encoder_outputs = tf.transpose(encoder_outputs,
                                                         [1, 0, 2])

                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
                    attention_encoder_outputs, multiplier=hparams.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    enc_state, multiplier=hparams.beam_width)
                tiled_encoder_inputs_lengths = tf.contrib.seq2seq.tile_batch(
                    encoder_inputs_lengths, multiplier=hparams.beam_width)

                # Create an attention mechanism
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    hparams.num_units, tiled_encoder_outputs,
                    memory_sequence_length=tiled_encoder_inputs_lengths)

                wrapped_de_cell = tf.contrib.seq2seq.AttentionWrapper(
                    dec_cell, attention_mechanism,
                    attention_layer_size=hparams.num_units)

                dec_initial_state = wrapped_de_cell.zero_state(
                    dtype=tf.float32,
                    batch_size=dynamic_batch_size * hparams.beam_width)
                dec_initial_state = dec_initial_state.clone(
                    cell_state=tiled_encoder_final_state)
        else:
            with tf.variable_scope(scope, reuse=True):
                wrapped_de_cell = dec_cell
                dec_initial_state = tf.contrib.seq2seq.tile_batch(
                    enc_state,
                    multiplier=hparams.beam_width)

        # len(inferred_reply) is lte encoder_length,
        # because we are targeting tweet (140 for each tweet)
        # Also by doing this,
        # we can pass the reply to other seq2seq w/o shorten it.
        maximum_iterations = hparams.encoder_length

        inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=wrapped_de_cell,
            embedding=embedding_encoder,
            start_tokens=tf.fill([dynamic_batch_size], hparams.sos_id),
            end_token=hparams.eos_id,
            initial_state=dec_initial_state,
            beam_width=hparams.beam_width,
            output_layer=projection_layer,
            length_penalty_weight=0.0)

        # Dynamic decoding
        with tf.variable_scope(scope, reuse=True):
            beam_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder, maximum_iterations=maximum_iterations)
        beam_replies = beam_outputs.predicted_ids
        return beam_replies

    @staticmethod
    def _build_sample_inference(hparams, embedding_encoder, enc_state,
                                enc_inputs_lengths, enc_outputs,
                                dec_cell, projection_layer, scope):
        # Sample decoder
        dynamic_batch_size = tf.shape(enc_inputs_lengths)[0]
        inference_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            embedding_encoder,
            tf.fill([dynamic_batch_size], hparams.sos_id), hparams.eos_id,
            softmax_temperature=0.1)  # 1.0 is default

        infer_logits, replies = ChatbotModel._dynamic_decode(dec_cell,
                                                             dynamic_batch_size,
                                                             enc_inputs_lengths,
                                                             enc_outputs,
                                                             enc_state,
                                                             hparams,
                                                             inference_helper,
                                                             projection_layer,
                                                             scope)
        return infer_logits, replies

    @staticmethod
    def _dynamic_decode(dec_cell, dynamic_batch_size,
                        enc_inputs_lengths, enc_outputs, enc_state,
                        hparams, dec_helper, projection_layer, scope):
        initial_state, wrapped_dec_cell = ChatbotModel._attention_wrapper(
            dec_cell, dynamic_batch_size, enc_inputs_lengths, enc_outputs,
            enc_state, hparams, scope)
        with tf.variable_scope(scope):
            with tf.variable_scope("infer"):
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    wrapped_dec_cell, dec_helper, initial_state,
                    output_layer=projection_layer)
        # len(inferred_reply) is lte encoder_length,
        # because we are targeting tweet (140 for each tweet)
        # Also by doing this,
        # we can pass the reply to other seq2seq w/o shorten it.
        maximum_iterations = hparams.encoder_length
        # Dynamic decoding
        # Here we reuse Attention Wrapper
        with tf.variable_scope(scope, reuse=True):
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                inference_decoder, maximum_iterations=maximum_iterations)
        replies = outputs.sample_id
        # We use infer_logits instead of _logits when calculating log_prob,
        # because infer_logits doesn't require decoder_target_lengths input.
        infer_logits = outputs.rnn_output
        return infer_logits, replies

    @staticmethod
    def _attention_wrapper(dec_cell, dynamic_batch_size, enc_inputs_lengths,
                           enc_outputs, enc_state, hparams, scope, reuse=True):
        # See https://github.com/tensorflow/tensorflow/issues/11904
        if hparams.use_attention:
            with tf.variable_scope(scope, reuse=reuse):
                # Attention
                # encoder_outputs is time major, so transopse it to batch major.
                # attention_encoder_outputs: [batch_size, encoder_length,
                # num_units]
                attention_encoder_outputs = tf.transpose(enc_outputs,
                                                         [1, 0, 2])

                # Create an attention mechanism
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    hparams.num_units,
                    attention_encoder_outputs,
                    memory_sequence_length=enc_inputs_lengths)

                wrapped_dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                    dec_cell, attention_mechanism,
                    attention_layer_size=hparams.num_units)

                initial_state = wrapped_dec_cell.zero_state(
                    dynamic_batch_size,
                    tf.float32).clone(
                    cell_state=enc_state)
        else:
            with tf.variable_scope(scope, reuse=reuse):
                wrapped_dec_cell = dec_cell
                initial_state = enc_state
        return initial_state, wrapped_dec_cell

    # convert sampled_indices to indices for tf.gather_nd.
    @staticmethod
    def _convert_indices(sampled_indices):
        print_sampled_indices = tf.Print(sampled_indices,
                                         [tf.shape(sampled_indices)],
                                         message="sampled_indices")
        batch_size = tf.shape(print_sampled_indices)[0]
        dec_length = tf.shape(print_sampled_indices)[1]
        print_batch_size = tf.Print(batch_size, [batch_size, dec_length],
                                    message="(batch_size, dec_length)")
        first_indices = tf.tile(
            tf.expand_dims(tf.range(print_batch_size), dim=1),
            [1, dec_length])
        second_indices = tf.reshape(
            tf.tile(tf.range(dec_length), [print_batch_size]),
            [print_batch_size, dec_length])
        print_first_indices = tf.Print(first_indices, [tf.shape(first_indices),
                                                       tf.shape(
                                                           second_indices)],
                                       message="(first_indices, "
                                               "second_indices)")
        return tf.stack([print_first_indices, second_indices, sampled_indices],
                        axis=2)


class TrainDataSource:
    def __init__(self, source_path, hparams, vocab_path=None):
        Shell.download_file_if_necessary(source_path)
        generator = TrainDataGenerator(source_path=source_path,
                                       hparams=hparams)
        # generator.remove_generated()
        train_dataset, vocab, rev_vocab = generator.generate(vocab_path)
        # We don't use shuffle here, because we want to align two data source
        #  here.
        self.train_dataset = train_dataset.repeat()
        self.vocab_path = generator.vocab_path
        # todo(higepon): Use actual validation dataset.
        self.valid_dataset = train_dataset.repeat()
        self.vocab = vocab
        self.rev_vocab = rev_vocab


class Trainer:
    def __init__(self):
        self.loss_step = []
        self.val_losses = []
        self.reward_step = []
        self.reward_average = []
        self.last_saved_time = datetime.datetime.now()
        self.last_stats_time = datetime.datetime.now()
        self.num_stats_per = 20
        self._valid_tweets = ["おはようございます。寒いですね。", "さて帰ろう。明日は早い。", "今回もよろしくです。"]

    def train_rl(self, rl_hparams, seq2seq_hparams, backward_hparams,
                 seq2seq_source_path, rl_source_path, tweets=None,
                 should_clean_saved_model=False):
        if tweets is None:
            tweets = []
        pp("===== Train RL {} ====".format(seq2seq_source_path))
        now = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
        pp("{}_rl_test".format(now))
        pp("rl_hparams")
        print_hparams(rl_hparams)

        seq2seq_data_source = TrainDataSource(seq2seq_source_path, rl_hparams)
        rl_data_source = TrainDataSource(rl_source_path, rl_hparams)
        easy_tf_log.set_dir(rl_hparams.model_path)
        Shell.download_model_data_if_necessary(rl_hparams.model_path)
        device = self._available_device()

        if should_clean_saved_model:
            clean_model_path(rl_hparams.model_path)
        with tf.device(device):
            rl_model = self.create_model(rl_hparams)
            seq2seq_model = self.create_model(seq2seq_hparams)
            backward_model = self.create_model(backward_hparams)

        vocab = seq2seq_data_source.vocab
        rev_vocab = seq2seq_data_source.rev_vocab
        infer_helper_rl = InferenceHelper(rl_model, vocab, rev_vocab)

        graph = rl_model.sess.graph
        _ = tf.summary.FileWriter(rl_hparams.model_path, graph)
        last_saved_time = datetime.datetime.now()

        with graph.as_default():
            seq2seq_train_data_next = \
                seq2seq_data_source.train_dataset.make_one_shot_iterator(

                ).get_next()
            rl_train_data_next = \
                rl_data_source.train_dataset.make_one_shot_iterator().get_next()

            global_step = None
            for step in range(rl_hparams.num_train_steps):
                with delta("data_fetch_time", global_step) as _:
                    seq2seq_train_data = rl_model.sess.run(
                        seq2seq_train_data_next)
                    rl_train_data = rl_model.sess.run(rl_train_data_next)

                batch_size = rl_hparams.batch_size

                # Sample!
                with delta("sample_time", global_step) as _:
                    samples, _ = rl_model.sample(seq2seq_train_data[0],
                                                 seq2seq_train_data[1])

                # Calc 1/N_a * logP_seq2seq(a|p_i, q_i) for each sampled.
                with delta("calc_reward_s", global_step) as _:
                    reward_s = self.calc_reward_s(seq2seq_model,
                                                  seq2seq_train_data,
                                                  samples)

                # Calc 1/N_qi * logP_backward(qi|a)
                # TODO: Vectorized implementation here.
                with delta("calc_reward_qi", global_step) as _:
                    reward_qi = self.calc_reward_qi(backward_model,
                                                    rl_train_data, samples)

                reward = reward_s + reward_qi
                max_len = len(samples[0])
                reward_avg = np.sum(reward) / max_len / batch_size

                # standardize reward
                # don't shift mean (by RL tips)
                # reward -= np.mean(reward)
                reward /= (np.std(reward))

                self._print_log("reward_avg", reward_avg, step=global_step)
                with delta("calc_entropy", global_step) as _:
                    self._print_log("entropy",
                                    self.calc_policy_entropy(infer_helper_rl),
                                    global_step)

                if True:  # step % 5 == 0:
                    validation_tweets = [
                        "危うく子供を引きかけた……駐車場でバックしようとしてたら子供が走って来てた:(",
                        "鏡に写る自分の顔を見て思ったヤバい、痩せすぎて頰が…そこで一大決心！今夜からちゃんと食べる",
                        "エスカレーター乗る位置で関西帰ってきたな〜〜って実感します🤔"]
                    with delta("valid_infer", global_step) as _:
                        for t in validation_tweets:
                            infer_helper_rl.print_inferences(t)

                    # greedy results from RL rl_model
                    with delta("rl_infer", global_step) as _:
                        replies, _ = rl_model.infer(seq2seq_train_data[0],
                                                    seq2seq_train_data[1])

                    # This is for debug to see if probability of RL looks
                    # reasonable.
                    with delta("calc_rl_reward", global_step) as _:
                        reward_s_rl = self.calc_reward_s(
                            rl_model,
                            seq2seq_train_data,
                            replies)

                    with delta("calc_rl_reward_qi", global_step) as _:
                        reward_qi_rl = self.calc_reward_qi(backward_model,
                                                           rl_train_data,
                                                           replies)

                    with delta("seq2seq_infer", global_step) as _:
                        seq2seq_replies, _ = seq2seq_model.infer(
                            seq2seq_train_data[0],
                            seq2seq_train_data[1])

                    # This is for debug to see if reward_s looks reasonable.
                    with delta("calc_seq2seq_reward_s", global_step) as _:
                        reward_s_seq2seq = self.calc_reward_s(
                            seq2seq_model,
                            seq2seq_train_data,
                            seq2seq_replies)
                    with delta("calc_seq2seq_reward_qi", global_step) as _:
                        reward_qi_seq2seq = self.calc_reward_qi(backward_model,
                                                                rl_train_data,
                                                                seq2seq_replies)

                    for batch in range(2):
                        pp(
                            infer_helper_rl.ids_to_string(
                                seq2seq_train_data[0][:, batch]))
                        pp(
                            "    [seq2] : {} {:.2f} => ({:.2f}) <= {"
                            ":.2f}".format(
                                infer_helper_rl.ids_to_string(
                                    seq2seq_replies[batch]),
                                reward_s_seq2seq[batch][0].item(),
                                reward_s_seq2seq[batch][0].item() +
                                reward_qi_seq2seq[batch][
                                    0].item(),
                                reward_qi_seq2seq[batch][0].item()))
                        pp(
                            "    [RL greedy] : {} {:.2f} => ({:.2f}) <= {"
                            ":.2f}".format(
                                infer_helper_rl.ids_to_string(replies[batch]),
                                reward_s_rl[batch][0].item(),
                                reward_s_rl[batch][0].item() +
                                reward_qi_rl[batch][0].item(),
                                reward_qi_rl[batch][0].item()))
                        pp(
                            "    [RL sample]: {} {:.2f} => ({:.2f}) <= {"
                            ":.2f}".format(
                                infer_helper_rl.ids_to_string(samples[batch]),
                                reward_s[batch][0].item(),
                                reward_s[batch][0].item() + reward_qi[batch][
                                    0].item(),
                                reward_qi[batch][0].item()))

                rl_hparams = rl_model.hparams
                with delta("train_with_reward", global_step) as _:
                    global_step, loss = rl_model.train_with_reward(
                        seq2seq_train_data[0],
                        seq2seq_train_data[1],
                        samples,
                        reward)
                    self._print_log("rl_loss", loss, global_step)
                if step != 0 and step % 100 == 0:
                    pp("save and restore")
                    rl_model.save()
                    is_restored = rl_model.restore()
                    assert is_restored
                    self._print_inferences(step, tweets, infer_helper_rl)
                    now = datetime.datetime.now()
                    pp("delta:", (now - last_saved_time).total_seconds())
                    last_saved_time = now
                    assert is_restored
                    self._save_model_in_drive(rl_hparams)
                    pp("step={}, global_step={}".format(step, global_step))

    #
    # Calculate action entropy.
    #
    # In general entropy is defined as -E[logP(X)] which is
    #  -Sum(p(X)logP(X)). But we can't calculate it, because we can't
    # enumerate all the
    # possible actions (= replies). Because (A) it's gonna be dec_len^(
    # vocab_size) pattern. (B) We can't list all the possible input to the
    # model. Here we calculate the entropy by limiting target beam_width = 3 and
    #  limiting # of input to 1.
    @staticmethod
    def calc_policy_entropy(infer_helper):
        tweets = ["おはようございます。今日も暑いですね",
                  "鏡に写る自分の顔を見て思ったヤバい、痩せすぎて頰が…そこで一大決心！今夜からちゃんと食べる",
                  "同じく寒かったので*はだいぶ楽になりました🙇💦᷆"]
        entropy = 0.0
        for tweet in tweets:
            encoder_inputs, encoder_inputs_lengths = \
                infer_helper.create_inference_input(
                    tweet)
            beam_replies, _, probs, log_probs = \
                infer_helper.model.infer_beam_search(
                    encoder_inputs, encoder_inputs_lengths)

            # [dec_len, vocab_size]
            log_prob = log_probs[0]
            prob = probs[0]
            # [dec_len, beam_size]
            replies = beam_replies[0]

            for i in range(infer_helper.model.hparams.beam_width):
                reply = replies[:, i]
                for idx, word_id in enumerate(reply):
                    if idx < len(log_prob):
                        entropy = entropy + log_prob[idx][word_id] * prob[idx][
                            word_id]
        return -entropy

    def calc_reward_qi(self, backward_model, train_data, samples):
        hparams = backward_model.hparams
        batch_size = hparams.batch_size
        max_len = len(samples[0])
        pp("reward_qi size=", batch_size, max_len)
        reward_qi = np.zeros((batch_size, max_len))
        # target label with eos.
        # [batch_size, dec_length]
        qi = train_data[2]
        a_enc_inputs, a_enc_inputs_lengths = self.format_enc_inputs(
            hparams, backward_model, samples)
        # [batch_size, dec_len, vocab_size]
        log_probs, _ = backward_model.log_probs(a_enc_inputs,
                                                a_enc_inputs_lengths)
        for batch in range(batch_size):
            tweet = qi[batch]
            tweet_len = 0
            p = 0
            for i, word_id in enumerate(tweet):
                # log_probs shape is supposed to be [batch_size,
                # dec_length, vocab_size],
                # but it sometimes becomes [batch_size,
                # smaller_value, vocab_size].
                # This is because we're using GreedyDecoder,
                # dynamic_decode finishes the decoder process when it
                #  sees eos_id.
                # If all enc_inputs ends up shorter dec_output,
                # we can have smaller_value here.
                if i < len(log_probs[batch]):
                    p += log_probs[batch][i][word_id]
                tweet_len = tweet_len + 1
                if word_id == hparams.eos_id:
                    break
            assert (tweet_len != 0)
            p /= tweet_len
            # reward is zero, after eos. So that we can ignore them.
            for i in range(min([tweet_len, max_len])):
                reward_qi[batch][i] = p
        return reward_qi

    @staticmethod
    def calc_reward_s(seq2seq_model, train_data, samples):
        max_len = len(samples[0])
        # [batch_size, dec_len]
        log_probs_sampled, logits1 = seq2seq_model.log_probs_sampled(
            train_data[0],
            train_data[1],
            samples)
        # log_probs_sampled2, logits2 = seq2seq_model.log_probs_sampled(
        #     train_data[0],
        #     train_data[1],
        #     samples)
        #
        # for b in range(seq2seq_model.hparams.batch_size):
        #     for i in range(max_len):
        #         for v in range(seq2seq_model.hparams.vocab_size):
        #             if logits1[b][i][v] != logits2[b][i][v]:
        #                 pp("Unmatch b={} i={} v={} {} vs {}".format(b,
        # i, v,
        #
        # logits1[
        #                                                                    b][
        #                                                                    i][
        #                                                                    v],
        #
        # logits2[
        #                                                                    b][
        #                                                                    i][
        #
        # v]))

        # if np.array_equal(log_probs_sampled, log_probs_sampled2):
        #     pp("log probs equl")
        # else:
        #     pp("noooo")

        # [batch_size, dec_len, vocab_size]
        # log_probs, _ = seq2seq_model.log_probs(train_data[0], train_data[1])
        # for batch in range(seq2seq_model.hparams.batch_size):
        #     log_probs_sampled_batch = log_probs_sampled[batch]
        #     for i in range(max_len):
        #         pp("debugging[{}][{}] {} {} = {}".format(batch, i,
        #
        # log_probs_sampled_batch[
        #                                                         i] ==
        #                                                     log_probs[
        # batch][i][
        #                                                         samples[
        # batch][
        #                                                             i]],
        #
        # log_probs_sampled_batch[
        #                                                         i],
        #                                                     log_probs[
        # batch][i][
        #                                                         samples[
        # batch][
        #                                                             i]]))

        batch_size = seq2seq_model.hparams.batch_size
        reward_s = np.zeros((batch_size, max_len))
        for batch in range(batch_size):
            tweet = samples[batch]
            tweet_len = 0
            p = 0
            for i in range(len(tweet)):
                p += log_probs_sampled[batch][i]
                tweet_len = tweet_len + 1
                if tweet[i] == seq2seq_model.hparams.eos_id:
                    break
            assert (tweet_len != 0)
            p /= tweet_len
            # reward is zero, after eos. So that we can ignore them.
            for i in range(tweet_len):
                reward_s[batch][i] = p
        return reward_s

    @staticmethod
    def format_enc_inputs(hparams, model, replies):
        enc_inputs = []
        enc_inputs_lengths = []

        # replies: [batch_size, dec_length]
        for reply in replies:
            reply_len = model.seq_len(reply.tolist())
            # Safe guard: sampled reply has sometimes 0 len.
            #            adjusted_len = hparams.encoder_length if reply_len
            # == 0 else reply_len
            enc_inputs_lengths.append(reply_len)
            if reply_len <= hparams.encoder_length:
                padded_reply = np.append(reply, ([hparams.pad_id] * (
                    hparams.encoder_length - len(reply))))
                enc_inputs.append(padded_reply)
            else:
                raise Exception(
                    "Inferred"
                    " reply shouldn't be longer than encoder_input")

        # Expected enc_inputs param is time major.
        enc_inputs = np.transpose(np.array(enc_inputs))
        return enc_inputs, enc_inputs_lengths

    @staticmethod
    def _reward_for_test(model, sampled_replies):
        max_len = len(sampled_replies[0])
        # default negative reward
        reward = np.ones((model.hparams.batch_size, max_len)) * -1.0
        good_value = 0
        for i, reply in enumerate(sampled_replies):
            reply_len = model.input_length(reply.tolist())
            if reply_len == 8 or reply_len == 0 or reply_len == 1:
                for r in range(max_len):
                    reward[i][r] = -1.0
            else:
                good_value += 1
                for r in range(max_len):
                    reward[i][r] = 1.0
        return good_value, reward

    def train_seq2seq_swapped(self, hparams, tweets_path, validation_tweets,
                              should_clean_saved_model=True, vocab_path=None):
        Shell.download_file_if_necessary(tweets_path)
        swapped_path = TrainDataGenerator.generate_source_target_swapped(
            tweets_path)
        return self.train_seq2seq(hparams, swapped_path, validation_tweets,
                                  should_clean_saved_model, vocab_path)

    def train_seq2seq(self, hparams, tweets_path, val_tweets,
                      should_clean_saved_model=True, vocab_path=None):
        pp("===== Train Seq2Seq {} ====".format(tweets_path))
        print_hparams(hparams)

        if should_clean_saved_model:
            clean_model_path(hparams.model_path)
        data_source = TrainDataSource(tweets_path, hparams, vocab_path)
        return self._train_loop(data_source, hparams, val_tweets)

    def _print_inferences(self, global_step, tweets, helper, ):
        pp("==== {} ====".format(global_step))
        len_array = []
        for tweet in tweets:
            len_array.append(len(helper.inferences(tweet)[0]))
            helper.print_inferences(tweet)
        self._print_log('average reply len', np.mean(len_array))

    @staticmethod
    def create_model(hparams):

        # See https://www.tensorflow.org/tutorials/using_gpu
        # #allowing_gpu_memory_growth
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        train_graph = tf.Graph()
        train_sess = tf.Session(graph=train_graph, config=config)
        with train_graph.as_default():
            with tf.variable_scope('root'):
                model = ChatbotModel(train_sess, hparams,
                                     model_path=hparams.model_path)
                if not model.restore():
                    train_sess.run(tf.global_variables_initializer())

        return model

    def _train_loop(self, data_source,
                    hparams, tweets):
        Shell.download_model_data_if_necessary(hparams.model_path)

        device = self._available_device()
        with tf.device(device):
            model = self.create_model(hparams)

        def my_train(**kwargs):
            data = kwargs['train_data']
            return model.train(data[0], data[1], data[2], data[3], data[4])

        return self._generic_train_loop(data_source, hparams,
                                        model,
                                        tweets, my_train)

    @staticmethod
    def _available_device():
        device = '/cpu:0'
        if has_gpu0():
            device = '/gpu:0'
            pp("$$$ GPU ENABLED $$$")
        return device

    @staticmethod
    def tokenize(infer_helper, text):
        tagger = MeCab.Tagger("-Owakati")
        words = tagger.parse(text).split()
        return infer_helper.words_to_ids(words)

    def _generic_train_loop(self, data_source, hparams,
                            model,
                            tweets, train_func):
        try:
            return self._raw_train_loop(data_source, hparams, model, train_func,
                                        tweets)
        except KeyboardInterrupt as ke:
            raise ke
        except Exception as e:
            pb.push_note("Train error", str(e))
            raise e

    def _raw_train_loop(self, data_source, hparams,
                        model, train_func,
                        tweets):
        vocab = data_source.vocab
        rev_vocab = data_source.rev_vocab
        infer_helper = InferenceHelper(model, vocab, rev_vocab)
        graph = model.sess.graph
        with graph.as_default():
            train_data_next = \
                data_source.train_dataset.make_one_shot_iterator().get_next()
            val_data_next = data_source.valid_dataset.make_one_shot_iterator(

            ).get_next()
            easy_tf_log.set_dir(hparams.model_path)
            writer = tf.summary.FileWriter(hparams.model_path, graph)
            self.last_saved_time = datetime.datetime.now()
            for i in range(hparams.num_train_steps):
                train_data = model.sess.run(train_data_next)

                step, summary = train_func(
                    train_data=train_data,
                )
                writer.add_summary(summary, step)

                if i != 0 and i % self.num_stats_per == 0:
                    model.save(hparams.model_path)
                    is_restored = model.restore()
                    assert is_restored
                    self._print_inferences(step, tweets, infer_helper)
                    self._compute_val_loss(step, model, val_data_next, writer)
                    #                    self._print_stats(hparams,
                    # learning_rate)
                    self._plot_if_necessary()
                    self._save_model_in_drive(hparams)
                else:
                    print('.', end='')
        return model, infer_helper

    def _plot_if_necessary(self):
        if len(self.reward_average) > 0 and len(self.reward_average) % 30 == 0:
            self._plot(self.reward_step, self.reward_average,
                       y_label='reward average')
            self._plot(self.loss_step, self.val_losses,
                       y_label='validation_loss')

    def _print_stats(self, hparams, learning_rate):
        pp("learning rate", learning_rate)
        delta = (
                    datetime.datetime.now() -
                    self.last_stats_time).total_seconds() * 1000
        self._print_log("msec/data",
                        delta / hparams.batch_size / self.num_stats_per)
        self.last_stats_time = datetime.datetime.now()

    def _save_model_in_drive(self, hparams):
        now = datetime.datetime.now()
        delta_in_min = (now - self.last_saved_time).total_seconds() / 60

        if delta_in_min >= 60:
            self.last_saved_time = datetime.datetime.now()
            Shell.save_model_in_drive(hparams.model_path)

    @staticmethod
    def _log(key, value, step=None):
        tflog("{}[{}]".format(key, current_client_id), value, step)

    @staticmethod
    def _print_log(key, value, step=None):
        tflog("{}[{}]".format(key, current_client_id), value, step)
        pp("{}={}".format(key, round(value, 1)))

    @staticmethod
    def _plot(x, y, x_label="step", y_label='y'):
        title = "{}_{}".format(current_client_id, y_label)
        plt.plot(x, y, label=title)
        plt.plot()
        plt.ylabel(title)
        plt.xlabel(x_label)
        plt.legend()
        plt.show()

    def _compute_val_loss(self, global_step, model, val_data_next,
                          writer):
        val_data = model.sess.run(val_data_next)
        val_loss, val_loss_log = model.batch_loss(val_data[0],
                                                  val_data[1],
                                                  val_data[2],
                                                  val_data[3],
                                                  val_data[4])
        # np.float64 to native float
        val_loss = val_loss.item()
        writer.add_summary(val_loss_log, global_step)
        self._print_log("validation loss", val_loss)
        self.loss_step.append(global_step)
        self.val_losses.append(val_loss)
        return val_loss


class InferenceHelper:
    def __init__(self, model, vocab, rev_vocab):
        self.model = model
        self.vocab = vocab
        self.rev_vocab = rev_vocab

    def inferences(self, tweet):
        encoder_inputs, encoder_inputs_lengths = self.create_inference_input(
            tweet)
        replies, _ = self.model.infer(encoder_inputs, encoder_inputs_lengths)
        ids = replies[0].tolist()
        all_infer = [self.sanitize_text(self.ids_to_words(ids))]
        beam_replies, logits, _, _ = self.model.infer_beam_search(
            encoder_inputs,
            encoder_inputs_lengths)
        beam_infer = [
            self.sanitize_text(self.ids_to_words(beam_replies[0][:, i])) for i
            in range(self.model.hparams.beam_width)]
        all_infer.extend(beam_infer)
        return all_infer

    @staticmethod
    def sanitize_text(line):
        line = re.sub(r"\[EOS\]", " ", line)
        line = re.sub(r"\[UNK\]", "💩", line)
        return line

    def print_inferences(self, tweet):
        pp(tweet)
        for i, reply in enumerate(self.inferences(tweet)):
            pp("    [{}]{}".format(i, reply))

    def words_to_ids(self, words):
        ids = []
        for word in words:
            if word in self.vocab:
                ids.append(self.vocab[word])
            else:
                ids.append(self.model.hparams.unk_id)
        return ids

    def ids_to_string(self, ids):
        return self.sanitize_text(self.ids_to_words(ids))

    def ids_to_words(self, ids):
        words = ""
        for word_id in ids:
            words += self.rev_vocab[word_id]
        return words

    def create_inference_input(self, text):
        inference_encoder_inputs = np.empty(
            (self.model.hparams.encoder_length, self.model.hparams.batch_size),
            dtype=np.int)
        inference_encoder_inputs_lengths = np.empty(
            self.model.hparams.batch_size, dtype=np.int)
        text = TrainDataGenerator.sanitize_line(text)
        tagger = MeCab.Tagger("-Owakati")
        words = tagger.parse(text).split()
        ids = self.words_to_ids(words)
        ids = ids[:self.model.hparams.encoder_length]
        len_ids = len(ids)
        ids.extend([self.model.hparams.pad_id] * (
            self.model.hparams.encoder_length - len(ids)))
        for i in range(self.model.hparams.batch_size):
            inference_encoder_inputs[:, i] = np.array(ids, dtype=np.int)
            inference_encoder_inputs_lengths[i] = len_ids
        return inference_encoder_inputs, inference_encoder_inputs_lengths


class ConversationTrainDataGenerator:
    def __init__(self):
        return

    # Generate the following file from conversations_txt file.
    # Let p_i:   line 3i     in the txt file, which is original tweet.
    #     q_i:   line 3i + 1 in the txt file, which is reply to the tweet.
    #     p_i+1: line 3i + 2 in the txt file, which is reply to the reply above.
    # (A) conversation_seq2seq.txt for train p_seq2seq and p_seq2seq_backward
    #     line 2i: p_i + q_i
    #     line 2i+1: p_i+1
    #
    # (B) conversation_rl.txt for train p_rl.
    #     line 2i: p_i + q_i
    #     line 2i+1: q_i
    #
    # (A) and (B) should share the vocabulary.
    # noinspection PyUnusedLocal
    def generate(self, conversations_txt):
        basename, extension = os.path.splitext(conversations_txt)
        seq2seq_path = "{}_seq2seq{}".format(basename, extension)
        rl_path = "{}_rl{}".format(basename, extension)
        with open(seq2seq_path, "w") as s_out, open(rl_path,
                                                    "w") as r_out, gfile.GFile(
            conversations_txt,
            mode="rb") as fin:
            tweet = None
            reply = None
            reply2 = None
            for i, line in enumerate(fin):
                line = line.decode('utf-8')
                line = line.rstrip()
                if i % 3 == 0:
                    tweet = line
                elif i % 3 == 1:
                    reply = line
                else:
                    reply2 = line
                    self._write(s_out, tweet, reply, reply2)
                    self._write(r_out, tweet, reply, reply)

    @staticmethod
    def _write(s_out, tweet, reply, reply2):
        s_out.write(tweet)
        s_out.write(' ')
        s_out.write(reply)
        s_out.write('\n')
        s_out.write(reply2)
        s_out.write('\n')


class TrainDataGenerator:
    def __init__(self, source_path, hparams):
        self.source_path = source_path
        self.hparams = hparams
        basename, extension = os.path.splitext(self.source_path)
        self.enc_path = "{}_enc{}".format(basename, extension)
        self.dec_path = "{}_dec{}".format(basename, extension)
        self.enc_idx_path = "{}_enc_idx{}".format(basename, extension)
        self.dec_idx_path = "{}_dec_idx{}".format(basename, extension)
        self.dec_idx_eos_path = "{}_dec_idx_eos{}".format(basename, extension)
        self.dec_idx_sos_path = "{}_dec_idx_sos{}".format(basename, extension)
        self.dec_idx_len_path = "{}_dec_idx_len{}".format(basename, extension)

        self.enc_idx_padded_path = "{}_enc_idx_padded{}".format(basename,
                                                                extension)
        self.enc_idx_len_path = "{}_enc_idx_len{}".format(basename, extension)

        self.vocab_path = "{}_vocab{}".format(basename, extension)

        self.generated_files = [self.enc_path, self.dec_path, self.enc_idx_path,
                                self.dec_idx_path, self.dec_idx_eos_path,
                                self.dec_idx_sos_path, self.dec_idx_len_path,
                                self.enc_idx_padded_path, self.vocab_path,
                                self.enc_idx_len_path]
        self.max_vocab_size = hparams.vocab_size
        self.start_vocabs = [hparams.sos_token, hparams.eos_token,
                             hparams.pad_token, hparams.unk_token]
        self.tagger = MeCab.Tagger("-Owakati")

    def remove_generated(self):
        for file in self.generated_files:
            if os.path.exists(file):
                os.remove(file)

    def generate(self, vocab_path=None):
        pp("generating enc and dec files...")
        self._generate_enc_dec()
        pp("generating vocab file...")
        if vocab_path is None:
            self._generate_vocab()
        else:
            shutil.copyfile(vocab_path, self.vocab_path)
        pp("loading vocab...")
        vocab, _ = self._load_vocab()
        pp("generating id files...")
        self._generate_id_file(self.enc_path, self.enc_idx_path, vocab)
        self._generate_id_file(self.dec_path, self.dec_idx_path, vocab)
        pp("generating padded input file...")
        self._generate_enc_idx_padded(self.enc_idx_path,
                                      self.enc_idx_padded_path,
                                      self.enc_idx_len_path,
                                      self.hparams.encoder_length)
        pp("generating dec eos/sos files...")
        self._generate_dec_idx_eos(self.dec_idx_path, self.dec_idx_eos_path,
                                   self.hparams.decoder_length)
        self._generate_dec_idx_sos(self.dec_idx_path, self.dec_idx_sos_path,
                                   self.dec_idx_len_path,
                                   self.hparams.decoder_length)
        pp("done")
        return self._create_dataset()

    def _generate_id_file(self, source_path, dest_path, vocab):
        if gfile.Exists(dest_path):
            return
        with gfile.GFile(source_path, mode="rb") as file, gfile.GFile(dest_path,
                                                                      mode="wb") as of:
            for line in file:
                line = line.decode('utf-8')
                words = self.tagger.parse(line).split()
                ids = [vocab.get(w, self.hparams.unk_id) for w in words]
                of.write(" ".join([str(word_id) for word_id in ids]) + "\n")

    def _load_vocab(self):
        rev_vocab = []
        with gfile.GFile(self.vocab_path, mode="r") as file:
            rev_vocab.extend(file.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            # Dictionary of (word, idx)
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab

    def _generate_vocab(self):
        if gfile.Exists(self.vocab_path):
            return
        vocab_dic = self._build_vocab_dic(self.enc_path)
        vocab_dic = self._build_vocab_dic(self.dec_path, vocab_dic)
        vocab_list = self.start_vocabs + sorted(vocab_dic, key=vocab_dic.get,
                                                reverse=True)
        if len(vocab_list) > self.max_vocab_size:
            pp("vocab_len=", len(vocab_list))
            vocab_list = vocab_list[:self.max_vocab_size]
        with gfile.GFile(self.vocab_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")

    # noinspection PyUnusedLocal,PyUnusedLocal
    def _generate_enc_dec(self):
        if gfile.Exists(self.enc_path) and gfile.Exists(self.dec_path):
            return
        with gfile.GFile(self.source_path, mode="rb") as file, gfile.GFile(
            self.enc_path, mode="w+") as ef, gfile.GFile(self.dec_path,
                                                         mode="w+") as df:
            tweet = None
            reply = None
            for i, line in enumerate(file):
                line = line.decode('utf-8')
                line = self.sanitize_line(line)
                if i % 2 == 0:
                    tweet = line
                else:
                    reply = line
                    if tweet and reply:
                        ef.write(tweet)
                        df.write(reply)
                    tweet = None
                    reply = None

    def _generate_enc_idx_padded(self, source_path, dest_path, dest_len_path,
                                 max_line_len):
        if gfile.Exists(dest_path):
            return
        with open(source_path) as fin, open(dest_path,
                                            "w") as fout, open(dest_len_path,
                                                               "w") as flen:
            line = fin.readline()
            while line:
                ids = [int(x) for x in line.split()]
                if len(ids) > max_line_len:
                    ids = ids[:max_line_len]
                    # i don't remember why we did this
                #                    ids = ids[-max_line_len:]
                flen.write(str(len(ids)))
                flen.write("\n")
                if len(ids) < max_line_len:
                    ids.extend(
                        [self.hparams.pad_id] * (max_line_len - len(ids)))
                ids = [str(x) for x in ids]
                fout.write(" ".join(ids))
                fout.write("\n")
                line = fin.readline()

    # read decoder_idx file and append eos at the end of idx list.
    def _generate_dec_idx_eos(self, source_path, dest_path, max_line_len):
        if gfile.Exists(dest_path):
            return
        with open(source_path) as fin, open(dest_path, "w") as fout:
            line = fin.readline()
            while line:
                ids = [int(x) for x in line.split()]
                if len(ids) > max_line_len - 1:
                    ids = ids[:max_line_len - 1]
                #                  ids = ids[-(max_line_len - 1):]
                ids.append(self.hparams.eos_id)
                if len(ids) < max_line_len:
                    ids.extend(
                        [self.hparams.pad_id] * (max_line_len - len(ids)))
                ids = [str(x) for x in ids]
                fout.write(" ".join(ids))
                fout.write("\n")
                line = fin.readline()

    # read decoder_idx file and put sos at the beginning of the idx list.
    # also write out length of index list.
    def _generate_dec_idx_sos(self, source_path, dest_path, dest_len_path,
                              max_line_len):
        if gfile.Exists(dest_path):
            return
        with open(source_path) as fin, open(dest_path, "w") as fout, open(
            dest_len_path, "w") as flen:
            line = fin.readline()
            while line:
                ids = [self.hparams.sos_id]
                ids.extend([int(x) for x in line.split()])
                if len(ids) > max_line_len:
                    ids = ids[:max_line_len]
                flen.write(str(len(ids)))
                flen.write("\n")
                if len(ids) < max_line_len:
                    ids.extend(
                        [self.hparams.pad_id] * (max_line_len - len(ids)))
                ids = [str(x) for x in ids]
                fout.write(" ".join(ids))
                fout.write("\n")
                line = fin.readline()

    @staticmethod
    def sanitize_line(line):
        # replace @username
        # replacing @username had bad impact where USERNAME token shows up
        # everywhere.
        #        line = re.sub(r"@([A-Za-z0-9_]+)", "USERNAME", line)
        line = re.sub(r"@([A-Za-z0-9_]+)", "", line)
        # Remove URL
        line = re.sub(r'https?://.*', "", line)
        line = line.lstrip()
        return line

    @staticmethod
    def generate_source_target_swapped(source_path):
        basename, extension = os.path.splitext(source_path)
        dest_path = "{}_swapped{}".format(basename, extension)
        with gfile.GFile(source_path, mode="rb") as fin, gfile.GFile(dest_path,
                                                                     mode="w+") as fout:
            temp = None
            for i, line in enumerate(fin):
                if i % 2 == 0:
                    temp = line
                else:
                    fout.write(line)
                    fout.write(temp)
                    temp = None
        return dest_path

    def _build_vocab_dic(self, source_path, vocab_dic=None):
        if vocab_dic is None:
            vocab_dic = {}
        with gfile.GFile(source_path, mode="r") as file:
            for line in file:
                words = self.tagger.parse(line).split()
                for word in words:
                    if word in vocab_dic:
                        vocab_dic[word] += 1
                    else:
                        vocab_dic[word] = 1
            return vocab_dic

    @staticmethod
    def _read_file(source_path):
        file = open(source_path)
        data = file.read()
        file.close()
        return data

    def _read_vocab(self, source_path):
        rev_vocab = []
        rev_vocab.extend(self._read_file(source_path).splitlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab

    def text_line_split_dataset(self, filename):
        return tf.data.TextLineDataset(filename).map(self.split_to_int_values)

    @staticmethod
    def split_to_int_values(x):
        return tf.string_to_number(tf.string_split([x]).values, tf.int32)

    def _create_dataset(self):

        tweets_dataset = self.text_line_split_dataset(self.enc_idx_padded_path)
        tweets_lengths_dataset = tf.data.TextLineDataset(
            self.enc_idx_len_path)

        replies_sos_dataset = self.text_line_split_dataset(
            self.dec_idx_sos_path)
        replies_eos_dataset = self.text_line_split_dataset(
            self.dec_idx_eos_path)
        replies_sos_lengths_dataset = tf.data.TextLineDataset(
            self.dec_idx_len_path)

        tweets_transposed = tweets_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(
                self.hparams.batch_size)).map(
            lambda x: tf.transpose(x))
        tweets_lengths = tweets_lengths_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(self.hparams.batch_size))

        replies_with_eos_suffix = replies_eos_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(self.hparams.batch_size))
        replies_with_sos_prefix = replies_sos_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(
                self.hparams.batch_size)).map(
            lambda x: tf.transpose(x))
        replies_with_sos_suffix_lengths = replies_sos_lengths_dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(
                self.hparams.batch_size))
        vocab, rev_vocab = self._read_vocab(self.vocab_path)
        return tf.data.Dataset.zip((tweets_transposed, tweets_lengths,
                                    replies_with_eos_suffix,
                                    replies_with_sos_prefix,
                                    replies_with_sos_suffix_lengths)), vocab, \
               rev_vocab


def print_hparams(hparams):
    result = {}
    for key in ['machine', 'batch_size', 'num_units', 'num_layers',
                'vocab_size',
                'embedding_size', 'learning_rate', 'learning_rate_decay',
                'use_attention', 'encoder_length', 'decoder_length',
                'max_gradient_norm', 'beam_width', 'num_train_steps',
                'model_path']:
        result[key] = hparams.get(key)
    pp("hparams=", result)


# Helper functions to test
def make_test_training_data(hparams):
    train_encoder_inputs = np.empty(
        (hparams.encoder_length, hparams.batch_size), dtype=np.int)
    train_encoder_inputs_lengths = np.empty(hparams.batch_size, dtype=np.int)
    training_target_labels = np.empty(
        (hparams.batch_size, hparams.decoder_length), dtype=np.int)
    training_decoder_inputs = np.empty(
        (hparams.decoder_length, hparams.batch_size), dtype=np.int)

    # We keep first tweet to validate inference.
    first_tweet = None

    for i in range(hparams.batch_size):
        # Tweet
        tweet = np.random.randint(low=0, high=hparams.vocab_size,
                                  size=hparams.encoder_length)
        train_encoder_inputs[:, i] = tweet
        train_encoder_inputs_lengths[i] = len(tweet)
        # Reply
        #   Note that low = 2, as 0 and 1 are reserved.
        reply = np.random.randint(low=2, high=hparams.vocab_size,
                                  size=hparams.decoder_length - 1)

        training_target_label = np.concatenate(
            (reply, np.array([hparams.eos_id])))
        training_target_labels[i] = training_target_label

        training_decoder_input = np.concatenate(([hparams.sos_id], reply))
        training_decoder_inputs[:, i] = training_decoder_input

        if i == 0:
            first_tweet = tweet
    return first_tweet, train_encoder_inputs, train_encoder_inputs_lengths, \
           training_target_labels, training_decoder_inputs


def test_training(hparams, model):
    if hparams.use_attention:
        pp("==== training model[attention] ====")
    else:
        pp("==== training model ====")
    first_tweet, train_encoder_inputs, train_encoder_inputs_lengths, \
    training_target_labels, training_decoder_inputs = make_test_training_data(
        hparams)
    for i in range(hparams.num_train_steps):
        _ = model.train(train_encoder_inputs,
                        train_encoder_inputs_lengths,
                        training_target_labels,
                        training_decoder_inputs,
                        np.ones(hparams.batch_size,
                                dtype=int) * hparams.decoder_length)
        if i % 5 == 0 and hparams.debug_verbose:
            print('.', end='')

        if i % 15 == 0:
            model.save()

    inference_encoder_inputs = np.empty((hparams.encoder_length, 1),
                                        dtype=np.int)
    inference_encoder_inputs_lengths = np.empty(1, dtype=np.int)
    for i in range(1):
        inference_encoder_inputs[:, i] = first_tweet
        inference_encoder_inputs_lengths[i] = len(first_tweet)

    # testing 
    log_prob54 = model.log_prob(inference_encoder_inputs,
                                inference_encoder_inputs_lengths,
                                np.array([5, 4]))
    log_prob65 = model.log_prob(inference_encoder_inputs,
                                inference_encoder_inputs_lengths,
                                np.array([6, 5]))
    pp("log_prob for 54", log_prob54)
    pp("log_prob for 65", log_prob65)

    reward = model.reward_ease_of_answering(hparams.encoder_length,
                                            inference_encoder_inputs,
                                            inference_encoder_inputs_lengths,
                                            np.array([[5], [6]]))
    pp("reward=", reward)

    if hparams.debug_verbose:
        pp(inference_encoder_inputs)
    replies, _ = model.infer(inference_encoder_inputs,
                             inference_encoder_inputs_lengths)
    pp("Inferred replies", replies[0])
    pp("Expected replies", training_target_labels[0])


def test_distributed_pattern(hparams):
    for d in [hparams.model_path]:
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    pp('==== test_distributed_pattern[{} {}] ===='.format(
        'attention' if hparams.use_attention else '',
        'beam' if hparams.beam_width > 0 else ''))

    first_tweet, train_encoder_inputs, train_encoder_inputs_lengths, \
    training_target_labels, training_decoder_inputs = make_test_training_data(
        hparams)

    model = Trainer().create_model(hparams)

    for i in range(hparams.num_train_steps):
        _ = model.train(train_encoder_inputs,
                        train_encoder_inputs_lengths,
                        training_target_labels,
                        training_decoder_inputs,
                        np.ones(hparams.batch_size,
                                dtype=int) * hparams.decoder_length)

    model.save()

    inference_encoder_inputs = np.empty(
        (hparams.encoder_length, hparams.batch_size),
        dtype=np.int)
    inference_encoder_inputs_lengths = np.empty(hparams.batch_size,
                                                dtype=np.int)

    for i in range(hparams.batch_size):
        inference_encoder_inputs[:, i] = first_tweet
        inference_encoder_inputs_lengths[i] = len(first_tweet)

    model.restore()
    replies, _ = model.infer(inference_encoder_inputs,
                             inference_encoder_inputs_lengths)
    pp("Inferred replies", replies[0])

    beam_replies, logits, _, _ = model.infer_beam_search(
        inference_encoder_inputs,
        inference_encoder_inputs_lengths)

    pp("logits", logits[0])
    pp("Inferred replies candidate0", beam_replies[0][:, 0])
    pp("Inferred replies candidate1", beam_replies[0][:, 1])

    inference_encoder_inputs = np.empty(
        (hparams.encoder_length, hparams.batch_size),
        dtype=np.int)
    inference_encoder_inputs_lengths = np.empty(hparams.batch_size,
                                                dtype=np.int)

    for i in range(hparams.batch_size):
        inference_encoder_inputs[:, i] = first_tweet
        inference_encoder_inputs_lengths[i] = len(first_tweet)

    replies = model.sample(inference_encoder_inputs,
                           inference_encoder_inputs_lengths)
    pp("sample replies", replies[0])
    pp("Expected replies", training_target_labels[0])


def test_distributed_one(enable_attention):
    hparams = copy.deepcopy(base_hparams).override_from_dict({
        'model_path': ModelDirectory.test_distributed.value,
        'use_attention': enable_attention,
        'beam_width': 2,
        'num_train_steps': 100,
        'learning_rate': 0.5
    })
    test_distributed_pattern(hparams)


def clean_model_path(model_path):
    shutil.rmtree(model_path)
    os.makedirs(model_path)


def print_header(text):
    pp("============== {} ==============".format(text))


def test_tweets_small_swapped(hparams):
    replies = ["@higepon おはようございます！", "おつかれさまー。気をつけて。", "こちらこそよろしくお願いします。"]
    trainer = Trainer()
    trainer.train_seq2seq_swapped(hparams, "tweets_small.txt", replies)


def test_tweets_large(hparams):
    tweets = ["さて福岡行ってきます！", "誰か飲みに行こう", "熱でてるけど、でもなんか食べなきゃーと思ってアイス買おうとしたの",
              "今日のドラマ面白そう！", "お腹すいたー", "おやすみ～", "おはようございます。寒いですね。",
              "さて帰ろう。明日は早い。", "今回もよろしくです。", "ばいとおわ！"]
    trainer = Trainer()
    trainer.train_seq2seq(hparams, "tweets_conversation.txt", tweets,
                          should_clean_saved_model=False)
    return trainer.model


def test_tweets_large_swapped(hparams):
    tweets = ["今日のドラマ面白そう！", "お腹すいたー", "おやすみ～", "おはようございます。寒いですね。",
              "さて帰ろう。明日は早い。", "今回もよろしくです。", "ばいとおわ！"]
    trainer = Trainer()
    trainer.train_seq2seq_swapped(hparams, "tweets_large.txt", tweets,
                                  should_clean_saved_model=False)
    return trainer.model


conversations_large_hparams = copy.deepcopy(base_hparams).override_from_dict(
    {
        # In typical seq2seq chatbot
        # num_layers=3, learning_rate=0.5, batch_size=64, vocab=20000-100000,
        #  learning_rate decay is 0.99, which is taken care as default
        # parameter in AdamOptimizer.
        'batch_size': 128,
        # of tweets should be dividable by batch_size default 64
        'encoder_length': 28,
        'decoder_length': 28,
        'num_units': 1024,
        'num_layers': 3,
        'vocab_size': 60000,
        # conversations.txt actually has about 70K uniq words.
        'embedding_size': 1024,
        'beam_width': 2,  # for faster iteration, this should be 10
        'num_train_steps': 0,
        'model_path': ModelDirectory.conversations_large.value,
        'learning_rate': 0.5,
        # For vocab_size 50000, num_layers 3, num_units 1024, tweet_large,
        # starting learning_rate 0.05 works well, change it t0 0.01 at
        # perplexity 800, changed it to 0.005 at 200.
        'learning_rate_decay': 0.99,
        'use_attention': True,

    })

# batch_size=128, learning_rage=0.001 work very well for RL. Loss decreases
# as expected. enthropy didn't flat out.

conversations_large_rl_hparams = copy.deepcopy(
    conversations_large_hparams).override_from_dict(
    {
        'model_path': ModelDirectory.conversations_large_rl.value,
        'num_train_steps': 2000,
        'learning_rate': 0.001,
        'beam_width': 3,
    })

conversations_large_backward_hparams = copy.deepcopy(
    conversations_large_hparams).override_from_dict(
    {
        'model_path': ModelDirectory.conversations_large_backward.value,
        'num_train_steps': 0,
    })


def test_train_rl():
    resume_rl = True

    conversations_txt = "conversations_large.txt"
    Shell.download_file_if_necessary(conversations_txt)
    ConversationTrainDataGenerator().generate(conversations_txt)

    trainer = Trainer()
    valid_tweets = ["さて福岡行ってきます！", "誰か飲みに行こう",
                    "熱でてるけど、でもなんか食べなきゃーと思ってアイス買おうとしたの",
                    "今日のドラマ面白そう！", "お腹すいたー", "おやすみ～", "おはようございます。寒いですね。",
                    "さて帰ろう。明日は早い。", "今回もよろしくです。", "ばいとおわ！"]
    trainer.train_seq2seq(conversations_large_hparams,
                          "conversations_large_seq2seq.txt",
                          valid_tweets, should_clean_saved_model=False)
    trainer.train_seq2seq_swapped(conversations_large_backward_hparams,
                                  "conversations_large_seq2seq.txt",
                                  ["この難にでも応用可能なひどいやつ",
                                   "おはようございます。明日はよろしくおねがいします。"],
                                  vocab_path="conversations_large_seq2seq_vocab.txt",
                                  should_clean_saved_model=False)

    if not resume_rl:
        sq.Shell.copy_saved_model(conversations_large_hparams,
                                  conversations_large_rl_hparams)
    Trainer().train_rl(conversations_large_rl_hparams,
                       conversations_large_hparams,
                       conversations_large_backward_hparams,
                       "conversations_large_seq2seq.txt",

                       "conversations_large_rl.txt",
                       valid_tweets)
