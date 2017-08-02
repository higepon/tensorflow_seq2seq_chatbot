import os
import sqlite3
import pickle
import tweepy
from datetime import datetime, timedelta

DB_NAME = 'tweets.db'


def create_tables():
    conn = sqlite3.connect(DB_NAME)
    sql = 'create table tweets(sid integer primary key, data blob not null, processed integer not null default 0)'
    c = conn.cursor()
    c.execute(sql)
    conn.commit()
    conn.close()
# alter table tweets add column bot_flag integer NOT NULL default 0;


def insert_tweet(status_id, tweet, bot_flag=0):
    conn = sqlite3.connect(DB_NAME)
    binary_data = pickle.dumps(tweet, pickle.HIGHEST_PROTOCOL)
    c = conn.cursor()
    c.execute("insert into tweets (sid, data, bot_flag) values (?, ?, ?)", [status_id, sqlite3.Binary(binary_data), bot_flag])
    conn.commit()
    conn.close()


class StreamListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        self.next_tweet_time = self.get_next_tweet_time()

    def on_status(self, status):
        print("{0}: {1}".format(status.text, status.author.screen_name))

        screen_name = status.author.screen_name
        # ignore my tweets
        if screen_name == self.api.me().screen_name:
            print("Ignored my tweet")
            return True
        elif status.text.startswith("@{0}".format(self.api.me().screen_name)):
            # Save mentions
            print("Saved mention")
            insert_tweet(status.id, status)
            return True
        else:
            if self.next_tweet_time < datetime.today():
                print("Saving normal tweet as seed")
                self.next_tweet_time = self.get_next_tweet_time()
                insert_tweet(status.id, status, bot_flag=1)
            print("Ignored this tweet")
            return True

    @staticmethod
    def get_next_tweet_time():
        return datetime.today() + timedelta(hours=2)

    @staticmethod
    def on_error(status_code):
        print(status_code)
        return True


def tweet_listener():
    consumer_key = os.getenv("consumer_key")
    consumer_secret = os.getenv("consumer_secret")
    access_token = os.getenv("access_token")
    access_token_secret = os.getenv("access_token_secret")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    while True:
        try:
            stream = tweepy.Stream(auth=api.auth,
                                   listener=StreamListener(api))
            print("listener starting...")
            stream.userstream()
        except Exception as e:
            print(e)
            print(e.__doc__)

if __name__ == '__main__':
    if False:
        create_tables()
    tweet_listener()
