import torch
import torchaudio
import torch.nn.functional as F


CLASSES = [
    'six', 'nine', 'on', 'left', 'three', 'five', 'go', 'bird', 'seven', 'off', 'wow', 'two', 'stop', 'zero', 'up', 'house', 'happy', 'cat', 'sheila', 'down', 'right', 'four', 'one', 'tree', 'eight', 'bed', 'marvin', 'dog', 'yes', 'no'
]


class SpeechRNN(torch.nn.Module):
  
    def __init__(self):
        super(SpeechRNN, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=13, hidden_size=256, 
            num_layers=2, batch_first=True
        )
        self.out_layer = torch.nn.Linear(256, 30)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        x = self.out_layer(out[:, -1, :])
        return self.softmax(x)


def create_mfcc(audio_path):
    waveform, fs = torchaudio.load(audio_path, normalization=True)

    # Convert to mono
    if waveform.size()[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Change sample rate to 16 KHz
    if fs != 16000:
        waveform = torchaudio.transforms.Resample(fs, 16000)(waveform)
    
    # Make sure that the audio is of atleast one second
    if waveform.shape[1] < 16000:
        waveform = F.pad(input=waveform, pad=(0, 16000 - waveform.shape[1]), mode='constant', value=0)
    
    return torchaudio.transforms.MFCC(n_mfcc=13, log_mels=True)(waveform).transpose(1, 2)


def get_prediction(audio_path, model):
    global CLASSES

    mfcc = create_mfcc(audio_path)
    print('mfcc shape:', mfcc.size())
    model.eval()
    output = model(mfcc).max(1)[1].item()
    print('output class:', output)

    return CLASSES[output]


def predict_text(audio_path, model_path):
    print('Loading model')
    model = SpeechRNN()
    model.load_state_dict(torch.load(model_path))
    print('Model loaded')

    return get_prediction(audio_path, model)
