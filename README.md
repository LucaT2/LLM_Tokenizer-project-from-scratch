# LLM_Tokenizer-project-from-scratch
## OVERVIEW

In this project I built a Tokenizer and an LLM from scratch for the purpose of understanding better how the Transformer arhitecture works. 
The model is available both in  my [Kaggle Notebook](https://www.kaggle.com/code/tasadanluca/llm-from-scratch) and in the python files from above. The version that is in the python files is a bit more clean, but the model is trained in kaggle as it allowed me to use a more powerful gpu so I was able to train it faster.
This is a decoder only Transformer, thus it generates random text based on the books that it has been trained on. 

## Data Collection
My training and validation data consists of about 25 early sci-fi books, in total the dataset has around 1.5 Milion words and after tokenization 2.2 Milion tokens. The books are downloaded from project Gutenberg. I cleaned the data by only keeping the content between a start and an end marker so as to not include header and footers that are not relevant to the content of the books and my purpose to generate text that looks somewhat like early Scifi.

## Tokenizer
My Tokenizer is trained using [BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding) (byte pair encoding) and produces 8256 tokens. I first split the text using the GPT4 split pattern using this regex `(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`, after which I encode it using UTF-8.

Afterwards I train it using BPE, merging the highest frequency bytes. Training this took around 8 hours so it's quite a time consuming process. After obtaining the 8000 new tokens I save them in two files: tokenizer.model and tokenizer.vocab, the .model file is for training the model and the .vocab file is just for visualizing and understanding the merges that the algorithm made.

## Transformer Model
My Transformer model is a decoder only model that generates text. It looks very similar to the figure below, with the mention that it does not take any input, but it generates new tokens based solely on what it has previously generated. Also I have Dropout layers after every other layer.


![Transformer Arhitecture](/Readme-assets/Decoder-only-model.jpg)


The dimension of the embedding in my model is 256, the multi-head layer consists of 6 heads, and I have 6 Transformer blocks. The context of the model is 256, so each token can learn from the 256 tokens that came before itself. I set my dropout to 0.2 and for normalization I use layer normalization. The activation function is Relu and this is applied only for the Feed Forward layer. The Feed Forward consists of two linear layers, a Relu layer between them and a Dropout layer at the end.

The last linear layer from the diagram is my modeling head, and this is responsible for computing the logits for the next token. The dimension of the output after this step will be a two dimensional matrix with each row representing a token, and each column a part of the embedding. I calculate the loss using [Cross Entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). In the forward pass I do no apply softmax as I need to compute only the loss. However, for generating, I apply a softmax to the output from the last linear layer to get the probabilities for each token. In order to generate the next token, instead of using the highest probable token, I sample from the distribution using [torch.multinomial](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html) in order to introduce more randomness to my generated text.

## Training
I will split my data into training data and validation data as I don't need test data due to the only purpose of the model being to generate text. In order for the train and validation split to be balanced, I take chunks of 20000 tokens and the last 10% of every chunk will be the validation data, the rest is the training data. Thus I ensure that both the validation data and the training data cover all the books. The optimizer that I use is [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) with the initial learning rate of 1e-3. Additionally, I have implemented a decaying learning rate every time the loss function plateaus, reducing the learing rate by a factor of 0.1 whenever the last 5 iterations do not decrease the loss substantially.

I use GPU for training with a batch size of 32. I train for 10000 iterations and I evaluate my loss at an interval of 100 iterations, each time on 200 batches. For generating, I do not give any input to the LLM, thus I initialize the initial context with zeros. I let it generate 500 tokens that I decode and then write them both to the output.txt and to the console. The training time just for the Transformer took somewhere between 75 and 90 minutes. There are 2 available files named model_weights1.pth and model_weights2.pth that contain already trained weights, so there is no need to retrain the model. The code in the repository corresponds to model_weights2. You can see the progress of the training iterations and the final train and validation losses in the kaggle notebook.
