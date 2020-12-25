# DialogueGCN
A pre-processing and training code for DialogueGCN on 
DailyDialogue and Mastodon dataset. 
Use Bert base to preprocess the sentences. 
Based on [DialogueGCN](https://github.com/declare-lab/conv-emotion/tree/master/DialogueGCN)

I have commited the changes to DialogueGCN repo.
If you still have any questions, welcome pr and issues!

## Paper Approach
See preprocess_dailydialog2.py and train_daily_feature.py 

Preprocess uses glove.840B.300d.txt to preprocess the InputSequence.

Training Use CNN to extract features from 300-dimension vector to 100-dimension vector.

The difference between train_daily_feature2.py and train_daily_feature3.py is the metrics.
We use the same settings in DialogRNN (see train_daily_feature3.py), which uses micro-f1 and 
masks the 'no-emotion' label.
Finally we get a f1 score about 44.24 on test set (50 epochs and select the final epoch results)

## Citation
If you use any source codes included in this repo in your work, please cite the following paper. 
The bibtex are listed below:

<pre>
@inproceedings{ghosal2019dialoguegcn,
  title={DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation},
  author={Ghosal, Deepanway and Majumder, Navonil and Poria, Soujanya and Chhaya, Niyati and Gelbukh, Alexander},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={154--164},
  year={2019}
}
@misc{qin2020cogat,
      title={Co-GAT: A Co-Interactive Graph Attention Network for Joint Dialog Act Recognition and Sentiment Classification}, 
      author={Libo Qin and Zhouyang Li and Wanxiang Che and Minheng Ni and Ting Liu},
      year={2020},
      eprint={2012.13260},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>