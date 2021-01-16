# -*- coding: utf-8 -*-
from chalicelib.param import *
from chalicelib.new_class import *
from chalicelib.func import *
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

data_transform = transforms.Compose([ 
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),   # Using ImageNet norms 
                         (0.229, 0.224, 0.225))]) # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2

vocab=load_obj('vocab')
# RNN models loading
rnn_decoder_5='rnn-decoder-5.pt'
rnn_encoder_5='rnn-encoder-5.pt'

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
encoder.load_state_dict(torch.load(drive_path + rnn_encoder_5,map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(drive_path + rnn_decoder_5,map_location=torch.device('cpu')))
rnn_model_5=[encoder, decoder]


im=Image.open("/content/test.jpg")

#FOR RNN
encoder, decoder=rnn_model_5
encoder.eval()
with torch.no_grad():
    image=data_transform(im)
    #image=image.to(device)
    image=image.unsqueeze_(0)
    features = encoder(image)
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sampled_caption=sampled_caption[1:-1]
    sentence = ' '.join(sampled_caption)
    sentence=sentence+'.'
    generated_caption=sentence.capitalize()