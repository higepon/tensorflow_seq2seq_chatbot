import os
import tensorflow as tf
import train
import datetime
import data_processer
import config
import tweepy
import time
import predict
import sqlite3
import pickle

consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_token_secret = os.getenv("access_token_secret")

DB_NAME = 'tweets.db'

def create_tables():
    conn = sqlite3.connect(DB_NAME)
    sql = 'create table tweets(sid integer primary key, data blob not null, processed integer not null default 0)'
    c = conn.cursor()
    c.execute(sql)
    conn.close()

def insert_tweet(status_id, tweet):
    conn = sqlite3.connect(DB_NAME)
    binary_data = pickle.dumps(mention, pickle.HIGHEST_PROTOCOL)
    c = conn.cursor()
    sqlite3.Binary(binary_data)
    c.execute("insert into tweets (sid, data) values (?, ?)", [status_id, sqlite3.Binary(binary_data)])
    conn.commit()
    conn.close()

def select_next_tweet():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("select sid, data from tweets where processed = 0")
    for row in c:
        sid = row[0]
        data = pickle.loads(row[1])
        return sid, data
    return None, None

def mark_tweet_processed(status_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("update tweets set processed = 1 where sid = ?", [status_id])
    conn.commit()
    conn.close()

def tweets():
    while True:
        status_id, tweet = select_next_tweet()
        if status_id is not None:
            yield(status_id, tweet)
        time.sleep(1)

def twitter_bot():
  # Only allocate part of the gpu memory when predicting.
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
  tf_config = tf.ConfigProto(gpu_options=gpu_options)

  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth)
  with tf.Session(config=tf_config) as sess:
    train.show_progress("Creating model...")
    model = train.create_or_restore_model(sess, train.buckets, forward_only=True)
    model.batch_size = 1
    train.show_progress("done\n")

    enc_vocab, _ = data_processer.initialize_vocabulary(config.VOCAB_ENC_TXT)
    _, rev_dec_vocab = data_processer.initialize_vocabulary(config.VOCAB_DEC_TXT)

    for tweet in tweets():
        status_id, status = tweet
        print("Processing {0}...".format(status.text))
        screen_name = status.author.screen_name        
        reply_body = predict.get_predition(sess,
                                           model,
                                           enc_vocab,
                                           rev_dec_vocab,
                                           status.text.encode('utf-8'))
        if reply_body is None:
            print("No reply predicted")
        else:
            reply_body = reply_body.replace('_UNK', '💩')
            reply_text = "@" + screen_name + " " + reply_body
            print("Reply:{0}".format(reply_text))
            api.update_status(status=reply_text,
                              in_reply_to_status_id=status_id)
        mark_tweet_processed(status_id)

if __name__ == '__main__':
  twitter_bot()
