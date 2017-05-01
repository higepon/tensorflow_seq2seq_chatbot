import config
import os
import sys
import math
import numpy as np
import tensorflow as tf
import data_processer
import lib.seq2seq_model as seq2seq_model

buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 60), (60, 70)]

def show_progress(text):
  sys.stdout.write(text)
  sys.stdout.flush()

def read_data_into_buckets(enc_path, dec_path, buckets):
  """Read tweets and reply and put them into buckets based on their length

  Args:
    enc_path: path to indexed tweets
    dec_path: path to indexed replies

  Returns:
    data_set: data_set[i] has [tweet, reply] pairs for bucket[i]
  """
  # data_set[i] corresponds data for buckets[i]
  data_set = [[] for _ in buckets]
  with tf.gfile.GFile(enc_path, mode="r") as ef, tf.gfile.GFile(dec_path, mode="r") as df:
      tweet, reply = ef.readline(), df.readline()
      counter = 0
      while tweet and reply:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in tweet.split()]
        target_ids = [int(x) for x in reply.split()]
        target_ids.append(data_processer.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(buckets):
          # Find bucket to put this conversation based on tweet and reply length
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        tweet, reply = ef.readline(), df.readline()
  return data_set


# Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
def create_or_restore_model(session, buckets, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel(config.MAX_ENC_VOCABULARY,
                                     config.MAX_DEC_VOCABULARY,
                                     buckets,
                                     config.LAYER_SIZE,
                                     config.NUM_LAYERS,
                                     config.MAX_GRADIENT_NORM,
                                     config.BATCH_SIZE,
                                     config.LEARNING_RATE,
                                     config.LEARNING_RATE_DECAY_FACTOR,
                                     forward_only=forward_only)

  print("model initialized")
  ckpt = tf.train.get_checkpoint_state(config.GENERATED_DIR)
  # the checkpoint filename has changed in recent versions of tensorflow
  checkpoint_suffix = ""
  if tf.__version__ > "0.12":
      checkpoint_suffix = ".index"
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def next_random_bucket_id(buckets_scale):
  n = np.random.random_sample()
  bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > n])
  return bucket_id

def train():
  # Only allocate 2/3 of the gpu memory to allow for running gpu-based predictions while training:
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
  tf_config = tf.ConfigProto(gpu_options=gpu_options)
  tf_config.gpu_options.allocator_type = 'BFC'


  with tf.Session(config=tf_config) as sess:

    show_progress("Setting up data set for each buckets...")
    train_set = read_data_into_buckets(config.TWEETS_TRAIN_ENC_IDX_TXT, config.TWEETS_TRAIN_DEC_IDX_TXT, buckets)
    valid_set = read_data_into_buckets(config.TWEETS_VAL_ENC_IDX_TXT, config.TWEETS_VAL_DEC_IDX_TXT, buckets)
    show_progress("done\n")

    show_progress("Creating model...")
    model = create_or_restore_model(sess, buckets, forward_only=False)
    show_progress("done\n")

    # list of # of data in ith bucket
    train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
    # This is for choosing randomly bucket based on distribution
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # Train Loop
    steps = 0
    previous_perplexities = []
    while True:
      bucket_id = next_random_bucket_id(train_buckets_scale)

      # Get batch
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
#      show_progress("Training bucket_id={0}...".format(bucket_id))

      # Train!
      _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
#      show_progress("done {0}\n".format(average_perplexity))


      steps = steps + 1
      if steps % 100 == 0:
        show_progress(".")
      if steps % 5000 != 0:
        continue

      # check point
      checkpoint_path = os.path.join(config.GENERATED_DIR, "seq2seq.ckpt")
      show_progress("Saving checkpoint...")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      show_progress("done\n")

      perplexity = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
      print ("global step %d learning rate %.4f perplexity "
             "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))

      # Decrease learning rate if no improvement was seen over last 3 times.
      if len(previous_perplexities) > 2 and perplexity > max(previous_perplexities[-3:]):
        sess.run(model.learning_rate_decay_op)
      previous_perplexities.append(perplexity)
      
      for bucket_id in range(len(buckets)):
        if len(valid_set[bucket_id]) == 0:
          print("  eval: empty bucket %d" % (bucket_id))
          continue
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(valid_set, bucket_id)
        _, average_perplexity, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        eval_ppx = math.exp(average_perplexity) if average_perplexity < 300 else float('inf')
        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

if __name__ == '__main__':
  train()
