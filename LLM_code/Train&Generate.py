from LLM import *
from utils_LLM import *
from Tokenizer import *
from Collect_data import *
from Tokenizer import Tokenizer

def tokenize_data(tokenizer:Tokenizer):
    MODEL_FILE = tokenizer.model_file
    VOCAB_FILE = tokenizer.vocab_file

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = f.read()
        print(f"Length of dataset: {len(data)}")
    
    if os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE):
        print(f"Model file '{MODEL_FILE}' and vocab file '{VOCAB_FILE}' already exist. Skipping training.")
    else:
        print("Training my tokenizer...(hope it works)")
        tokenizer.train(data)
        print("Tokenizer trained (hopefully) ")
        print("Saving tokenizer...")
        tokenizer.save()
        print("Tokenizer saved")

    print("Loading tokenizer")
    tokenizer.load()
    print("Tokenizer loaded")
    return data

    # print("Checking my tokenizer: ")
    # dummy_sentence = "I am confident my tokenizer works as it should"
    # print(f"Original sentence: {dummy_sentence}")
    #
    # encoded_dummy = tokenizer.encode(dummy_sentence)
    # print(f"Encoded sentence: {encoded_dummy}")
    #
    # decoded_dummy = tokenizer.decode(encoded_dummy)
    # print(f"Decoded sentence: {decoded_dummy}")
ENCODED_FILE = r"LLM_code/files/encoded_text"
def prepare_data(tokenizer:Tokenizer, data):
    if os.path.exists(ENCODED_FILE):
        print(f"Encoded file '{ENCODED_FILE}' already exist. Skipping encoding.")
        new_data = torch.load(ENCODED_FILE, weights_only=True)
    else:
        new_data = torch.tensor(tokenizer.encode(data), dtype=torch.long)
        torch.save(new_data, ENCODED_FILE)

    chunk_size = 20000
    split_percentage = int(0.9 * chunk_size)
    train_chunks = []
    validation_chunks = []

    for i in range(0, len(new_data), chunk_size):
        current_chunk = new_data[i:(i + chunk_size)]
        train_chunks.append(current_chunk[:split_percentage])
        validation_chunks.append(current_chunk[split_percentage:])

    train_data = torch.cat(train_chunks)
    validation_data = torch.cat(validation_chunks)

    print(f"Original data size: {len(new_data)}")
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(validation_data)}")
    return train_data, validation_data

def train_llm(train_data, validation_data):
    model = MyLLM()
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    # I will also add a decaying learning rate when the loss function plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        threshold=1e-3,
        threshold_mode='rel'

    )
    print(f"Initial learning rate: {learning_rate}")

    for iter in range(num_max_iterations):
        if iter % periodicity_of_evaluation == 0 or iter == num_max_iterations - 1:
            losses = approximate_loss(model, train_data, validation_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            scheduler.step(losses['val'])

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

        X, Y = get_batch('train', train_data, validation_data)

        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return m

# This function loads a model that has already been trained
def load_model_from_files():
    model = MyLLM()
    # There is also another 'pth' file with the weights from another model that
    # is similar to this one, but I believe this one is a bit better
    # This is also the model that corresponds to the code written in the .py files
    state_dict = torch.load('LLM_code/files/model_weights2.pth', weights_only = True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def generate_output(tokenizer, m):
    print("\n\n")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist())
    with open('output.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(generated_text)



def main():
    tokenizer  = Tokenizer()
    data = tokenize_data(tokenizer)
    train_data, validation_data = prepare_data(tokenizer, data)

    #If you do not want to train it you can load the pre-trained weights
    option = int(input("Choose if you want to train the model or load an already " \
    "trained one:\nEnter 0 for the trained one and 1 if you want to train it yourself: "))

    if option == 0:
        model = load_model_from_files()
    else:
        model = train_llm(train_data, validation_data)
    generate_output(tokenizer, model)

main()