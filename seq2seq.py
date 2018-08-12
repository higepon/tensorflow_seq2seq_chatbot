# mkdir -p ~/seq2seq_run; cd ~/seq2seq_run; python ~/Google\ Drive/tensorflow_seq2seq_chatbot/seq2seq.py

import lib.chatbot_model as sq
import copy as copy

conversations_large_hparams = copy.deepcopy(sq.base_hparams).override_from_dict(
    {
        # In typical seq2seq chatbot
        # num_layers=3, learning_rate=0.5, batch_size=64, vocab=20000-100000, learning_rate decay is 0.99, which is taken care as default parameter in AdamOptimizer.
        'batch_size': 128,  # of tweets should be dividable by batch_size default 64
        'encoder_length': 28,
        'decoder_length': 28,
        'num_units': 1024,
        'num_layers': 3,
        'vocab_size': 60000,
    # conversations.txt actually has about 70K uniq words.
        'embedding_size': 1024,
        'beam_width': 2,  # for faster iteration, this should be 10
        'num_train_steps': 0,
        'model_path': sq.ModelDirectory.conversations_large.value,
        'learning_rate': 0.5,
    # For vocab_size 50000, num_layers 3, num_units 1024, tweet_large, starting learning_rate 0.05 works well, change it t0 0.01 at perplexity 800, changed it to 0.005 at 200.
        'learning_rate_decay': 0.99,
        'use_attention': True,

    })

# batch_size=128, learning_rage=0.001 work very well for RL. Loss decreases as expected. enthropy didn't flat out.

conversations_large_rl_hparams = copy.deepcopy(
    conversations_large_hparams).override_from_dict(
    {
        'model_path': sq.ModelDirectory.conversations_large_rl.value,
        'num_train_steps': 2000,
        'learning_rate': 0.001,
        'beam_width': 3,
    })


conversations_large_backward_hparams = copy.deepcopy(
    conversations_large_hparams).override_from_dict(
    {
        'model_path': sq.ModelDirectory.conversations_large_backward.value,
        'num_train_steps': 0,        
    })

resume_rl = False

conversations_txt = "conversations_large.txt"
sq.Shell.download_file_if_necessary(conversations_txt)
sq.ConversationTrainDataGenerator().generate(conversations_txt)


trainer =sq.Trainer()
valid_tweets = ["さて福岡行ってきます！", "誰か飲みに行こう", "熱でてるけど、でもなんか食べなきゃーと思ってアイス買おうとしたの",
      "今日のドラマ面白そう！", "お腹すいたー", "おやすみ～", "おはようございます。寒いですね。",
      "さて帰ろう。明日は早い。", "今回もよろしくです。", "ばいとおわ！"]
trainer.train_seq2seq(conversations_large_hparams,
                      "conversations_large_seq2seq.txt",
                      valid_tweets, should_clean_saved_model=False)
trainer.train_seq2seq_swapped(conversations_large_backward_hparams,
                              "conversations_large_seq2seq.txt",
                              ["この難にでも応用可能なひどいやつ", "おはようございます。明日はよろしくおねがいします。"], vocab_path="conversations_large_seq2seq_vocab.txt", should_clean_saved_model=False)

if not resume_rl:
  sq.Shell.copy_saved_model(conversations_large_hparams, conversations_large_rl_hparams)
sq.Trainer().train_rl(conversations_large_rl_hparams,
                        conversations_large_hparams,
                        conversations_large_backward_hparams,
                        "conversations_large_seq2seq.txt",

                        "conversations_large_rl.txt",
                        valid_tweets)


