import pickle
import torch
import numpy as np

from attention import make_model


SRC_STOI = None
TARGET_ITOS = None
TRG_EOS_TOKEN = None
TRG_SOS_TOKEN = None


def load_metadata(meta_path):
    global SRC_STOI, TARGET_ITOS, TRG_EOS_TOKEN, TRG_SOS_TOKEN

    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    SRC_STOI = metadata['src_stoi']
    TARGET_ITOS = metadata['target_itos']
    TRG_EOS_TOKEN= metadata['trg_eos_index']
    TRG_SOS_TOKEN = metadata['trg_sos_index']


def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""
    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
    
    output = np.array(output)
    
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output
  

def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab[i] for i in x]

    return [str(t) for t in x]


def inference_logic(src, model):
    print('inference logic')
    hypotheses = [] 

    src_lengths = [len(src)]

    src_index = []
    for i in src:
        src_index.append(SRC_STOI[i])

    src = torch.LongTensor(src_index)
    src = src.unsqueeze(0)

    src_mask = torch.ones(src_lengths) > 0
    src_lengths = torch.LongTensor(src_lengths)

    pred = greedy_decode(
        model,
        src,
        src_mask,
        src_lengths,
        max_len=25,
        sos_index=TRG_SOS_TOKEN,
        eos_index=TRG_EOS_TOKEN,
    )

    hypotheses.append(pred)

    return hypotheses


def translate_sentence(german_sentence, weights_path, metadata_path):
    # Load model metadata
    print('loading metadata')
    load_metadata(metadata_path)

    # Load model
    global SRC_STOI, TARGET_ITOS
    print('loading model')
    model = make_model(weights_path, len(SRC_STOI), len(TARGET_ITOS))

    german_sentence = german_sentence.split()
    hypotheses = inference_logic(german_sentence, model)
    hypotheses = [lookup_words(x, TARGET_ITOS) for x in hypotheses]
    hypotheses = [" ".join(x) for x in hypotheses]
    return hypotheses[0]
