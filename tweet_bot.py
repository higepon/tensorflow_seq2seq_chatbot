# mkdir -p ~/seq2seq_run; cd ~/seq2seq_run; python ~/Google\
# Drive/tensorflow_seq2seq_chatbot/tweet_bot.py

import lib.chatbot_model as sq

sq.listener(sq.conversations_large_rl_hparams)
