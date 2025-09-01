import torch
from Hyperparameters import number_of_batches_to_eval, block_size, batch_size, device

def get_batch(split, train_data, validation_data):
  inter_data = train_data if split == 'train' else validation_data
  ix = torch.randint(len(inter_data) - block_size, (batch_size,))
  x = torch.stack([inter_data[i:i+block_size] for i in ix])
  y = torch.stack([inter_data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def approximate_loss(model, train_data, validation_data):
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(number_of_batches_to_eval)
    for k in range(number_of_batches_to_eval):
      X,Y = get_batch(split, train_data, validation_data)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out