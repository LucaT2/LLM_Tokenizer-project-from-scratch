import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#torch.manual_seed(1337)

#Hyperparameters
VOCAB_SIZE = 8256
NUM_MERGES = VOCAB_SIZE -256
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


block_size = 256
batch_size = 32
periodicity_of_evaluation = 100
learning_rate = 1e-3 #maybe change it depends on how the model performs
num_max_iterations = 10000
number_of_batches_to_eval = 200
dropout = 0.2
num_embeddings = 256
num_heads = 6
num_layers = 6

#devices = torch.get_all_devices()
#print([torch.cuda.device(i) for i in range(torch.cuda.device_count())])