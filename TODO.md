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
- 6/18 Re-read seq2seq model API https://www.tensorflow.org/tutorials/seq2seq
- 6/18 Improvement 1: Support Tensorboard
- 6/19 Try AdagradOptimizer and compare hom much faster it converges
- 6/19 Check if dataset is on memory
- Improvement 2: Construct the response in a non-greedy way
  - 6/24 Understand general beam search
  - 6/25 Read wikipedia
  - 6/25 Find actual tensorflow beam search example
  - 6/25 Make list of items to immplement
    - 6/25 wait, maybe implment in the graph is the right way?
    - 6/25 Understand current greedy model throughly?
- Beam Search ToDo
  - Search a bit to get better understanding
  - Understand current greedy model throughly
    - Do we really feed encoder input when beam search?
  - Decide it's a right way to implment it in the model
  - figure out how to do it
  - Will have to understand [this disucssion](https://github.com/tensorflow/tensorflow/issues/654)
  - Examnples
    - [demo](https://github.com/tensorflow/tensorflow/issues/654#issuecomment-168237741)
    - [seq2seq model](https://github.com/pbhatia243/Neural_Conversation_Models)
  - Naive implementation
  - Refactor switch between greedy and beam search  using flags
  - Evaluate 0.0.2
  - Merge
  - blog the result
- Clean up tweets data, we see a lot これw
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
### 0.0.2
beam search implemented.

    >おはよう
    normal:おはようございます
    beam
    0 おはよう ござい ます
    1 お は あり
    2 お は あり です 〜 ♪

    >こんにちは
    normal:はい(˘ω˘)
    beam
    0 はい ( ˘ ω ˘ )
    1 はい ( ˘ ω ˘ ) ｽﾔｧ
    2 はい ( ˙ㅿ˙ 。 .
    3 はい ♡
    4 はい ( ´ ω 。
    5 はい 、 さ www
    6 はい ( 笑

    >ばいばいー
    わろきちってんじゃんwww
    normal:beam
    0 がち やし ま ー ん
    1 いや ま ー ！
    2 わろ ぶ や ！
    3 ほら
    4 ネタ やし ぶ
    5 ど ま ー
    6 がち やし ま ーー
    7 いつの間に ま ー
    8 す
    9 いつの間に ぶ
    10 いつの間に やし ぶ うち
    11 やらかし た ❤
    12 現実 やし
    13 ほんま やし ぶ ()
    14 や ま ー

    >（月曜日から）逃げちゃ駄目だ……！
    normal;えぇこれは、、、
    beam
    0 なんで 進捗 は これ じゃ ねぇ ・ ・ ω ！
    1 え ぇ これ は 光 は 、 !
    2 え ぇ これ は 嫌 ）
    3 なんで 進捗 おっ け （ ω ！
    4 なんで 進捗 は これ じゃ ぞ 〜

    > 子供たちにつられて苦手なミニオンズ…(´･ω･`)w
    normal:気をしてねー(˘ω˘)
    beam
    0 気 を し て ( ˘ つ ω -(´∀｀; )
    1 気 を すん な ( ˙ ˘ )
    2 仕事 を すん や ( ˙ ω -(´∀｀; )
    3 気 を し て ねー 。 ( ^ ｰ ` ・ )
    4 気 を し てる やろ ( ˙ ˘ ω ˘ ･) !
    5 気 を し てる やろ ( ˙ ˘ ω ˘ ω ・ )
    6 気 を し てる の だ よ )

    > 中華そば醤油をいただきました💕お、おいしい〜😍大盛いけたかも？
    normal: 追加ですよねwww
    beam
    0 追加 し まし た ☺
    1 追加 です よ ☺ 
    2 追加 です よ ね www
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
