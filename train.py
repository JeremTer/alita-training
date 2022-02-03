import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader


from cnn import CNNetwork
from main import AlitaSoundDataset

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = 'datasets/audios.csv'
AUDIO_DIR = 'datasets'
SAMPLE_RATE = 44100
NUM_SAMPLES = 90000

labels = ['dance','wakeup','start','stop','weather','night','joke','leave','reminder']

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)


def index_to_label(param):
    return labels[param]


def predict(model,tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    print(tensor.shape)
    #tensor = tensor[None,None,:]
    print(tensor)
    tensor = model(tensor.unsqueeze(0))
    print("LES PREDICTIONS :")
    #print(tensor)
    tensor = get_likely_index(tensor)
    print("L'INDEX LE PLUS GRAND")
    #print(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    asd = AlitaSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES,mel_spectrogram)

    train_dataloader = create_data_loader(asd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNetwork().to(device)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    countSame = 0
    for i in range(len(asd)):
        signal, label = asd[i]
        result = predict(cnn, signal)
        print("JE PREDIS -> " + result)
        print("J'ETAIS -> " +  index_to_label(label))
        if result == index_to_label(label):
            countSame += 1

    print("RESULTAT FINAL = " + str(countSame) + " SUR " + str(len(asd)))

    # save model
    # torch.save(m5.state_dict(), "feedforwardnet.pth")
    # print("Trained feed forward net saved at feedforwardnet.pth")