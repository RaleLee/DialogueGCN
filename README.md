# DialogueGCN
A pre-processing and training code for DialogueGCN on 
DailyDialogue and Mastodon dataset. 
Use Bert base to preprocess the sentences. 
Based on [DialogueGCN](https://github.com/SenticNet/conv-emotion/tree/master/DialogueGCN)


Welcome pr and issues!

# Paper Approach
See preprocess_dailydialog2.py and train_daily_feature.py 

Preprocess uses glove.840B.300d.txt to preprocess the InputSequence.

Training Use CNN to extract features from 300-dimension vector to 100-dimension vector.

The difference between train_daily_feature2.py and train_daily_feature3.py is the metrics.
We use the same settings in DialogRNN (see train_daily_feature3.py), which uses micro-f1 and 
masks the 'no-emotion' label.
Finally we get a f1 score about 44.24 on test set (50 epochs and select the final epoch results)
