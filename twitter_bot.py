import os
import tensorflow as tf
import train
import datetime
import data_processer
import config
import tweepy
import time
import predict

consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_token_secret = os.getenv("access_token_secret")

class myExeption(Exception): pass

class StreamListener(tweepy.StreamListener):
    def __init__(self, api, sess, model, enc_vocab, rev_dec_vocab):
        self.api = api
        self.sess = sess
        self.model = model
        self.enc_vocab = enc_vocab
        self.rev_dec_vocab = rev_dec_vocab
        self.last_tweet_date_time = datetime.datetime.today()
    
    def on_status(self, status):
        print("{0}: {1}".format(status.text, status.author.screen_name))
        screen_name = status.author.screen_name

        # ignore my tweets
        if screen_name == self.api.me().screen_name:
            return True
        if status.text.startswith("@{0}".format(self.api.me().screen_name)):
            print("this is mention")
            status_id = status.id            
            reply_body = predict.get_predition(self.sess,
                                               self.model,
                                               self.enc_vocab,
                                               self.rev_dec_vocab,
                                               status.text.encode('utf-8'))
            if reply_body is not None:
                reply_body = reply_body.replace('_UNK', '💩')
                reply_text = "@" + screen_name + " " + reply_body
                self.api.update_status(status=reply_text,
                                       in_reply_to_status_id=status_id)
        else:
            # Other tweets in my timeline

            print("not a mention")
            if self.last_tweet_date_time + datetime.timedelta(hours=1) < datetime.datetime.today():            
#            if self.last_tweet_date_time + datetime.timedelta(minutes=3) < datetime.datetime.today():
                print("time to tweet")
                status_id = status.id            
                reply_body = predict.get_predition(self.sess,
                                                   self.model,
                                                   self.enc_vocab,
                                                   self.rev_dec_vocab,
                                                   status.text.encode('utf-8'))
                if reply_body is not None:                
                    reply_text = reply_body.replace('_UNK', '💩')
                    self.api.update_status(status=reply_text)
                    self.last_tweet_date_time = datetime.datetime.today()
            else:
                print("wait a bit")
        return True

    def on_error(self, status_code):
        print(status_code)
        return True

def twitter_bot():
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

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    while True:
        try:
            stream = tweepy.Stream(auth=api.auth,
                                   listener=StreamListener(api, sess, model, enc_vocab, rev_dec_vocab))
            stream.userstream()
        except Exception as e:
            print(e.__doc__)
            print(e.message)

if __name__ == '__main__':
  twitter_bot()
