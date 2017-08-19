import predict
import tensorflow as tf
import os
import json
import tweepy
import time
import socket
import http.client
from tweepy import OAuthHandler, Stream
from tweepy.streaming import StreamListener

tcpip_delay = 0.25
MAX_TCPIP_TIMEOUT = 16


class QueueListener(StreamListener):

    def __init__(self, sess):
        consumer_key = os.getenv("consumer_key")
        consumer_secret = os.getenv("consumer_secret")
        access_token = os.getenv("access_token")
        access_token_secret = os.getenv("access_token_secret")

        self.auth = OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)
        self.predictor = predict.EasyPredictor(sess)

    def on_data(self, data):
        """Routes the raw stream data to the appropriate method."""
        raw = json.loads(data)
        if 'in_reply_to_status_id' in raw:
            if self.on_status(raw) is False:
                return False
        elif 'limit' in raw:
            if self.on_limit(raw['limit']['track']) is False:
                return False
        return True

    def on_status(self, status):
        if 'retweeted_status' in status:
            return True
        text = status['text']
        replies = self.predictor.predict(text)
        if not replies:
            return True
        reply_body = replies[0]
        text = text.replace('\n', ' ')
        print(text)
        print("reply:{}".format(reply_body))
        return True

    def on_error(self, status):
        print('ON ERROR:', status)

    def on_limit(self, track):
        print('ON LIMIT:', track)


def main():
    with tf.Session() as sess:
        listener = QueueListener(sess)
        stream = Stream(listener.auth, listener)
        stream.filter(languages=["ja"],
                      track=['「', '」', '私', '俺', 'わたし', 'おれ', 'ぼく', '僕', 'http', 'www', 'co', '@', '#', '。', '，', '！',
                             '.', '!', ',', ':', '：', '』', ')', '...', 'これ'])
        try:
            while True:
                try:
                    stream.sample()
                except KeyboardInterrupt:
                    print('KEYBOARD INTERRUPT')
                    return
                except (socket.error, http.client.HTTPException):
                    global tcpip_delay
                    print('TCP/IP Error: Restarting after %.2f seconds.' % tcpip_delay)
                    time.sleep(min(tcpip_delay, MAX_TCPIP_TIMEOUT))
                    tcpip_delay += 0.25
        finally:
            stream.disconnect()
            print('Exit successful, corpus dumped in %s' % (listener.dumpfile))


if __name__ == '__main__':
    main()