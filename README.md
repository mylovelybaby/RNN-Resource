# RNN-Resource
## Tutorials
### General
- 动手学深度学习.in Chinese. includes basic rnn, gru, and lstm. https://zh.gluon.ai/chapter_recurrent-neural-networks/index.html
- The Unreasonable Effectiveness of RNNs by Andrej Karpathy. http://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Understanding LSTM Networks. Nice figures of LSTM cell.https://colah.github.io/posts/2015-08-Understanding-LSTMs/
![alt text](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

### Attention
- Attention and Memory: http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
- https://distill.pub/2016/augmented-rnns/#attentional-interfaces

### Beam Search
- attention and beam search: https://guillaumegenthial.github.io/sequence-to-sequence.html

### Perplexity
- exp(loss)

### RNN Dropout

### Bi-directional RNN
![alt text](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/bidirectional-rnn.png)
- http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

### Neural Turing Machine
- https://distill.pub/2016/augmented-rnns/#neural-turing-machines

### Neural Machine Translation:
- https://research.googleblog.com/2016/09/a-neural-network-for-machine.html
https://www.tensorflow.org/tutorials/seq2seq
- Stanford NMT group: https://nlp.stanford.edu/projects/nmt/

## Implementation
### Vanilla RNN in Python/Numpy:
- Minimal character-level language model with a Vanilla Recurrent Neural Network: https://gist.github.com/karpathy/d4dee566867f8291f086

### Vanilla RNN in Tensorflow
- General tensorflow tutorial. https://github.com/nlintz/TensorFlow-Tutorials and https://github.com/aymericdamien/TensorFlow-Examples
- Recurrent Neural Networks in Tensorflow. https://www.tensorflow.org/versions/master/tutorials/recurrent
- RNNs in Tensorflow, a Practical Guide and Undocumented Features: http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
- Vanishing Gradient: http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
- Char-rnn-tensorflow: https://github.com/sherjilozair/char-rnn-tensorflow

### Vanilla RNN in PyTorch:
- Word-level RNN PyTorch example: https://github.com/pytorch/examples/tree/master/word_language_model
- Practical PyTorch: https://github.com/spro/practical-pytorch
- NLP in PyTorch: https://github.com/rguthrie3/DeepLearningForNLPInPytorch

### Vanilla RNN in Theano
- GRU/LSTM RNN with Python and Theano: http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
- Word-level RNN with Python and theano: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

### Neural Machine Translation
- Neural Machine Translation (seq2seq) Tutorial https://www.tensorflow.org/tutorials/seq2seq https://github.com/tensorflow/nmt/
- https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate
- Baseline SMT data: http://www-lium.univ-lemans.fr/~schwenk/cslm_joint_paper/
- Baseline SMT paper: http://www-lium.univ-lemans.fr/~schwenk/papers/Sennrich.multidomain.acl2013.pdf

### Chatbot:
- stanford tensorflow tutorial:https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/2017/assignments/chatbot
- Deep QA: https://github.com/Conchylicultor/DeepQA
- chatbot-rnn reddit data: https://github.com/pender/chatbot-rnn
- summary/datasets: https://github.com/fendouai/Awesome-Chatbot
- in chinese: https://github.com/qhduan/Seq2Seq_Chatbot_QA

### Neural Turing Machine
- in tensorflow: https://github.com/carpedm20/NTM-tensorflow

### Visual Question Answering
- https://avisingh599.github.io/deeplearning/visual-qa/ https://github.com/avisingh599/visual-qa

### image to latex:
- https://guillaumegenthial.github.io/image-to-latex.html

### Text Summarization
- thunlp https://github.com/thunlp/TensorFlow-Summarization.git 
- textsum https://github.com/tensorflow/models/tree/master/research/textsum

## Paper:
- see https://github.com/kjw0612/awesome-rnn
![alt text](https://raw.githubusercontent.com/mylovelybaby/RNN-Resource/master/classic_seq2seq_paper.png)

## Datasets:
### Language model:
- Shakespeare: https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt
- Wikipedia: https://cs.stanford.edu/people/karpathy/char-rnn/wiki.txt
- Sherlock: https://sherlock-holm.es/stories/plain-text/cnus.txt
### Machine Translation
- WMT14: http://www.statmt.org/wmt14/translation-task.html
