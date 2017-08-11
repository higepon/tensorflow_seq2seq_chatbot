import os
import tensorflow as tf
import tweepy
import time
import predict
import sqlite3
import pickle
import tweet_listener


def select_next_tweet():
    conn = sqlite3.connect(tweet_listener.DB_NAME)
    c = conn.cursor()
    c.execute("select sid, data, bot_flag from tweets where processed = 0")
    for row in c:
        sid = row[0]
        data = pickle.loads(row[1])
        bot_flag = row[2]
        return sid, data, bot_flag
    return None, None, None


def mark_tweet_processed(status_id):
    conn = sqlite3.connect(tweet_listener.DB_NAME)
    c = conn.cursor()
    c.execute("update tweets set processed = 1 where sid = ?", [status_id])
    conn.commit()
    conn.close()


def tweets():
    while True:
        status_id, tweet, bot_flag = select_next_tweet()
        if status_id is not None:
            yield(status_id, tweet, bot_flag)
        time.sleep(1)


def post_reply(api, bot_flag, reply_body, screen_name, status_id):
    unk_count = reply_body.count('_UNK')
    reply_body = reply_body.replace('_UNK', '💩')
    if bot_flag == tweet_listener.SHOULD_TWEET:
        if unk_count > 0:
            return
        reply_text = reply_body
        print("My Tweet:{0}".format(reply_text))
        if not reply_text:
            return
        api.update_status(status=reply_text)
    else:
        if not reply_body:
            reply_body = "🐶(適切なお返事が生成できませんでした)"
        reply_text = "@" + screen_name + " " + reply_body
        print("Reply:{0}".format(reply_text))
        api.update_status(status=reply_text,
                          in_reply_to_status_id=status_id)


def twitter_bot():
    # Only allocate part of the gpu memory when predicting.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)

    consumer_key = os.getenv("consumer_key")
    consumer_secret = os.getenv("consumer_secret")
    access_token = os.getenv("access_token")
    access_token_secret = os.getenv("access_token_secret")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    with tf.Session(config=tf_config) as sess:
        predictor = predict.EasyPredictor(sess)

        for tweet in tweets():
            status_id, status, bot_flag = tweet
            print("Processing {0}...".format(status.text))
            screen_name = status.author.screen_name
            replies = predictor.predict(status.text)
            if not replies:
                print("no reply")
                continue
            reply_body = replies[0]
            if reply_body is None:
                print("No reply predicted")
            else:
                try:
                    post_reply(api, bot_flag, reply_body, screen_name, status_id)
                except tweepy.TweepError as e:
                    # duplicate status
                    if e.api_code == 187:
                        pass
                    else:
                        raise
            mark_tweet_processed(status_id)


if __name__ == '__main__':
    twitter_bot()
