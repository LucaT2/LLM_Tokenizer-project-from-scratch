# LLM_Tokenizer-project-from-scratch
## OVERVIEW

In this project I built a Tokenizer and an LLM from scratch for the purpose of understanding better how the Transformer arhitecture works. 
The model is available both in  my [Kaggle Notebook](https://www.kaggle.com/code/tasadanluca/llm-from-scratch) and in the python files from above. The version that is in the python files is a bit more clean, but the model is trained in kaggle as it allowed me to use a more powerful gpu so I was able to train it faster.
This is a decoder only Transformer, thus it generates random text based on the books that it has been trained on. 

## Data Collection
My training and validation data consists of about 25 early sci-fi books, in total the dataset has around 1.5 Milion words and after tokenization 2.2 Milion tokens. The books are downloaded from project Gutenberg. I cleaned the data by only keeping the content between a start and an end marker so as to not include header and footers that are not relevant to the content of the books and my purpose to generate text that looks somewhat like early scifi.

## Tokenizer
My Tokenizer is trained using [BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding) (byte pair encoding) and produces 8256 tokens. I first split the text using the GPT4 split pattern using this regex `(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+`, after which I encode it using UTF-8. Then I train it using BPE, mergin the highest frequency bytes. Training this took around 8 hours so it's quite a time consuming process. After obtaining the 8000 new tokens I save them in two files: tokenizer.model and tokenizer.vocab, the model file is for training the model and the vocab file is just for visualizing and understanding the merges that the algorithm made.

## Transformer Model
My Transformer model is a decoder only model that generates text. It looks very similar to the figure below, with the mention that it does not take any input, but it generates solely on what it has generated before.
[Transformer Arhitecture]
