import config
import numpy as np
import tensorflow as tf
import data_processer
import lib.seq2seq_model as seq2seq_model

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
def create_model(session, buckets, forward_only):

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
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

    print("Setting up data set for each buckets")
    train_set = read_data_into_buckets(config.TWEETS_TRAIN_ENC_IDX_TXT, config.TWEETS_TRAIN_DEC_IDX_TXT, buckets)
    print("Done")

    print("Creating model...")
    model = create_model(sess, buckets, forward_only=False)
    print("Done")

    # list of # of data in ith bucket
    train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # Originally from https://github.com/1228337123/tensorflow-seq2seq-chatbot
    # This is for choosing randomly bucket based on distribution
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # Train Loop
    while True:
      bucket_id = next_random_bucket_id(train_buckets_scale)

      # Get batch
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

      print("Training bucket_id={0}...".format(bucket_id))

      # Train!
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
    
      print("Done")


if __name__ == '__main__':
  train()
