# Goal
Make *fun* chatbot like human.
# TODO
- [done] Understand how CS20SI chatbot works
  - 6/11 read her documents
  - 6/11 train
  - 6/11 chat
  - 6/11 draw its model structure
    - didn't draw, but here is pretty similar one
  - 6/11 Post it to my blog
  - 6/11 Read improvement section of her document
  - 6/11 Make improvement plan here
- [done] 6/17 Make old chatbot compatible with tensorflow 1.0
  - 6/17 http://qiita.com/taroc/items/b9afd914432da08dafc8
  - 6/17 Run it with old seq2seq model
  - 6/17 Run it with new seq2seq model
  - 6/17 train with very small data
  - 6/17 predict
- Make basic chatbot
  - 6/11 Prepare pycharm env with python 3.3 and latest TensorFlow
  - 6/12 Port data.py to Python3
    - 6/12 fix O(1) slowness
    - 6/12 Compare generated file with the original ones.
  - 6/12 Investigate why data.py is so slow
  - 6/17 Write own train.py
    - 6/13 make an empty file
    - 6/13 commit it
    - 6/14 Read https://www.tensorflow.org/install/migration
    - 6/17 Make model
    - 6/17 Comment for each unknown lines
    - 6/17 write model function
    - 6/17  See if it's easy to port the old chatbot to 0.1.0 using https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
  - 6/17 Very naive implmentation with the same data as CS20SI
  - 6/17 Compare the two bots
  - 6/17 Probably rewrite it in [new seq2seq API](https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention)
- 6/18 Find a best way to record trial and error
  - 6/18 model parameters (do we have to keep all the parameter files?)
    - Yes, with directory in tag name
  - 6/18 How we construct model
  - 6/18 Maybe tag or release?
  - 6/18 Think what should be done while training? Read papers?
- 6/18 Re-read http://web.stanford.edu/class/cs20si/assignments/a3.pdf and make a list of todo again
- Re-read seq2seq model API https://www.tensorflow.org/tutorials/seq2seq
- Improvement 1: Support Tensorboard
- Try AdagradOptimizer and compare hom much faster it converges
- Check if dataset is on memory
- Improvement 2: Construct the response in a non-greedy way
- Improvement 3: Use more than just one utterance as the encoder
- Improvement 4: Create a chatbot with personality
- Improvement 5: Make your chatbot remember information from the previous conversation
- Improvements todo
  - Train on multiple datasets
  - Create a feedback loop that allows users to train your chatbot
- Make deploy process for tweet bot
- Make tweet bot available in cloud

# random ideas
- Can we use 2ch.net data?

## tags
### 0.0.1
Adam optimizer and summary op work well.
    global step 25000 learning rate 0.4522 perplexity 19.24
    eval: bucket 0 perplexity 4.63
    eval: bucket 1 perplexity 13.32
    eval: bucket 2 perplexity 27.23
    eval: bucket 3 perplexity 2.20
    > おはようー
    おはようございます
    > どうなの最近？
    これはいいの？
    > あほか。
    なにしwww
    > 君の名は
    まじでしょ？
    > まじかよ。
    とりあえず増えてないんですか
    > 最近映画見た？
    これ？
    > いやちがうよ。それじゃないって。適当なこと言うなよ。
    _UNKかwww
    > うんこじゃない
    どんな撮www
