# What is this?
This is seq2seq chatbot implementation. Most credit goes to [1228337123](https://github.com/1228337123/tensorflow-seq2seq-chatbot). I'm just reimplmenting their work to have better understandings on seq2seq. This chat bot is optimized for Japanese. You may change replace tokenizer with one for your language.

Main differences of my implmentation are
- More comments
- Easy to understand input/output format for each processes

# How to run
1. Prepare train data.
    1. Put your train data as data/tweets.txt, the file consists of pairs of tweet and reply.
    1. Odd lines are tweets and even lines are corresponding replies.
    1. You can get the training data using [github.com/Marsan-Ma/twitter_scraper](https://github.com/Marsan-Ma/twitter_scraper).
1. Process the training data and generate vocabulary file and some necessary files. Run following command then you'd see the files generated in generated/ directory.

    python data_processer.py
1. Train! Train may take a few hours to 1 day, and it never stops. Once you think it's ready, just Ctrl-C. Model parameters are saved in generated/ directory.

    python train.py
    
1. Talk to him!
    python predict.py
    
# Twitter Bot
By running twitter_bot.py, you can run this chatbot in twitter.
Here are some interesting conversations with him.
![sample1](http://cdn-ak.f.st-hatena.com/images/fotolife/h/higepon/20170428/20170428211132.jpg?1493381493?changed=1493381493)
![sample2](http://cdn-ak.f.st-hatena.com/images/fotolife/h/higepon/20170428/20170428211230.jpg?1493381551?changed=1493381551)
