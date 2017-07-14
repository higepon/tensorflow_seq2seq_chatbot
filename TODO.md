# Goal
Make *fun* chatbot like human.
# TODO
- __done__ 6/11 [Understand how CS20SI chatbot works](https://github.com/higepon/tensorflow_seq2seq_chatbot/wiki/Understand-how-CS20SI-chatbot-works)
- __done__ 6/17 [Make old chatbot compatible with tensorflow 1.0](https://github.com/higepon/tensorflow_seq2seq_chatbot/wiki/Make-old-chatbot-compatible-with-tensorflow-1.0)
- __done__ beam search
- Clean up tweets data, we see a lot これw
- Get 5M data
- [A] Improvement 3: Use more than just one utterance as the encoder
   - Check if tweet collector get one more deep conversation
   - collect
- Improvement 4: Create a chatbot with personality
- [B] Improvement 5: Make your chatbot remember information from the previous conversation
   - can we simpley train with 1 pair?
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
