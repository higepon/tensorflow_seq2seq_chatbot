import os
import sqlite3
import pickle
import tweepy

DB_NAME = 'tweets.db'

def create_tables():
    conn = sqlite3.connect(DB_NAME)
    sql = 'create table tweets(sid integer primary key, data blob not null, processed integer not null default 0)'
    c = conn.cursor()
    c.execute(sql)
    conn.commit()
    conn.close()

def insert_tweet(status_id, tweet):
    conn = sqlite3.connect(DB_NAME)
    binary_data = pickle.dumps(tweet, pickle.HIGHEST_PROTOCOL)
    c = conn.cursor()
    c.execute("insert into tweets (sid, data) values (?, ?)", [status_id, sqlite3.Binary(binary_data)])
    conn.commit()
    conn.close()

def select_next_tweet():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("select sid, data from tweets where processed = 0")
    for row in c:
        print(row)
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
    
consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_token_secret = os.getenv("access_token_secret")

class myExeption(Exception): pass

class StreamListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api

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
            print("Ignored this tweet")
            return True

    def on_error(self, status_code):
        print(status_code)
        return True

def tweet_listener():
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
            print(e.__doc__)

if __name__ == '__main__':
    if False:
        create_tables()
    tweet_listener()
