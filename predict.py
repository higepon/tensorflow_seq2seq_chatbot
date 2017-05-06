import sys
import tensorflow as tf
import numpy as np
import train
import config
import data_processer

def get_predition(session, model, enc_vocab, rev_dec_vocab, text):
  try:
    token_ids = data_processer.sentence_to_token_ids(text, enc_vocab)
    bucket_id = min([b for b in range(len(train.buckets))
                     if train.buckets[b][0] > len(token_ids)])
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    if data_processer.EOS_ID in outputs:
      outputs = outputs[:outputs.index(data_processer.EOS_ID)]
    return "".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
  except Exception as e:
    print(e)
    return None
  
def predict():
  # Only allocate part of the gpu memory when predicting.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
  tf_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(config=tf_config) as sess:
    train.show_progress("Creating model...")
    model = train.create_or_restore_model(sess, train.buckets, forward_only=True)
    model.batch_size = 1
    train.show_progress("done\n")

    enc_vocab, _ = data_processer.initialize_vocabulary(config.VOCAB_ENC_TXT)
    _, rev_dec_vocab = data_processer.initialize_vocabulary(config.VOCAB_DEC_TXT)

    sys.stdout.write("> ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    while line:
      line = line.encode('utf-8')
      predicted = get_predition(sess, model, enc_vocab, rev_dec_vocab, line)
      print(predicted)
      print("> ", end="")
      sys.stdout.flush()
      line = sys.stdin.readline()

if __name__ == '__main__':
  predict()
