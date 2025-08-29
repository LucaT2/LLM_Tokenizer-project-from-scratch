from Hyperparameters import *

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(num_embeddings, head_size, bias = False)
    self.query = nn.Linear(num_embeddings, head_size, bias = False)
    self.value = nn.Linear(num_embeddings, head_size, bias = False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask', torch.tril(torch.ones(block_size,block_size)))
  def forward(self, x): # x is the input
    num_batches, time_steps, channels = x.shape
    keys = self.key(x)
    queries = self.query(x)
    values = self.value(x)

    weights = queries @ keys.transpose(-2,-1) * channels**-0.5 # weights will be of size (B,T,T)
    weights = weights.masked_fill(self.mask[:time_steps,:time_steps] == 0, float('-inf'))
    weights = F.softmax(weights, dim = -1)
    weights = self.dropout(weights)
    output = weights @ values #output will be of size (B,T,C)
    return output

class MultiHead(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.projection = nn.Linear(n_heads * head_size, num_embeddings)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.projection(out))
    return out

class FeedForward(nn.Module):
  # This is a simple linear layer with a non-liner layer that follows it
  def __init__(self, n_embeddings):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embeddings, 4*n_embeddings),
        nn.ReLU(),
        nn.Linear(4*n_embeddings, n_embeddings),
        nn.Dropout(dropout)
    )
  def forward(self, x):
    return self.net(x)


class TransformerBlock(nn.Module):
  # This is a whole transformer block
  def __init__(self, n_embeddings, n_heads):
    super().__init__()
    head_size = n_embeddings // n_heads
    self.multihead = MultiHead(n_heads, head_size)
    self.feedforward = FeedForward(n_embeddings)
    self.layernorm1 = nn.LayerNorm(n_embeddings)
    self.layernorm2 = nn.LayerNorm(n_embeddings)

  def forward(self, x):
    x = x + self.multihead(self.layernorm1(x))
    x = x + self.feedforward(self.layernorm2(x))
    return x


class MyLLM(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(VOCAB_SIZE, num_embeddings)
    self.position_embedding_table = nn.Embedding(block_size, num_embeddings)
    self.blocks = nn.Sequential(*[TransformerBlock(num_embeddings, num_heads) for _ in range(num_layers)])
    self.layernorm = nn.LayerNorm(num_embeddings)
    self.modeling_head = nn.Linear(num_embeddings, VOCAB_SIZE)

  def forward(self,idx, targets = None):
    token_emb = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device = device))
    x = token_emb + pos_emb
    x = self.blocks(x)
    x = self.layernorm(x)
    logits = self.modeling_head(x)
    if targets is None:
      loss = None
    else:
      num_batches, time_steps, channels = logits.shape
      logits = logits.view(num_batches*time_steps, channels)
      targets = targets.view(num_batches*time_steps)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] # these are the idx on which the generate will depend upon
      logits, loss = self(idx_cond) # these are the predictions
      logits = logits[:, -1, :] # only last prediction matters
      probs = F.softmax(logits, dim = -1) # get the probabilities
      idx_next = torch.multinomial(probs, num_samples = 1) # sample from the distribution so the token generated can be other than the one with maximum probability
      idx = torch.cat((idx, idx_next), dim = 1) # append the new index to the running sequence
    return idx