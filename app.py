#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import logging

from chalice import Chalice
from chalice import BadRequestError

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage,ImageMessage
print('!!a!!')

from PIL import Image
from io import BytesIO
import boto3

print('!!b!!')

logger = logging.getLogger()
app = Chalice(app_name='line-api')
handler = WebhookHandler(os.environ.get('LINE_CHANNEL_SECRET'))
linebot = LineBotApi(os.environ.get('LINE_CHANNEL_ACCESS_TOKEN'))

print('!!c!!')

@app.route('/callback', methods=['POST'])
def callback():
    try:
        request = app.current_request

        # get X-Line-Signature header value
        signature = request.headers['x-Line-Signature']

        # get request body as text
        body = request.raw_body.decode('utf8')

        # handle webhook body
        handler.handle(body, signature)
    except Exception as err:
        logger.exception(err)
        raise BadRequestError('Invalid signature. Please check your channel access token/channel secret.')

    return 'OK'


def _valid_reply_token(event):
    '''
    Webhook のテスト時には reply token が 0 のみで構成されているので、
    その時に False を返します
    '''
    return not re.match('^0+$', event.reply_token)


@handler.add(MessageEvent, message=TextMessage)
def reply_for_text_message(event):
    ''' テキストメッセージを受け取った場合の応答 '''
    print('!!d!!')
    if not _valid_reply_token(event):
        return
    print('!!e!!')
    ans = "画像を送信してね。"
    print('!!f!!')
    reply_message(event, TextSendMessage(text=ans))
    print('!!g!!')

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print('!!h!!')
    print("handle_image:", event)

    try:
        print('!!i!!')
        s3 = boto3.client('s3', region_name='ap-northeast-1')
        print('!!j!!')
        bucket = 's3-nic-line-bot'
        key_dec = 'rnn-decoder-5.pt'
        key_enc = 'rnn-encoder-5.pt'
        print('!!k!!')
        response_dec = s3.get_object(Bucket=bucket, Key=key_dec)
        print('!!l!!')
        response_enc = s3.get_object(Bucket=bucket, Key=key_enc)
        print('!!m!!')
        content_dec = response_dec['Body']
        print('!!n!!')
        content_enc = response_enc['Body']
        print('!!o!!')
        print(content_dec)
        #print(content_dec.read())
        rnn_decoder_5_pt = BytesIO(content_dec.read())
        print('!!p!!')
        print(content_enc)
        #print(content_enc.read())
        rnn_encoder_5_pt = BytesIO(content_enc.read())
        print('!!q!!')
        message_id = event.message.id
        print('!!r!!')
        message_content = linebot.get_message_content(message_id)
        print('!!s!!')
        image = BytesIO(message_content.content)
        print('!!t!!')
        im = Image.open(image)
        print('!!u!!')

        from chalicelib.param import embed_size,hidden_size,num_layers,drive_path
        print('!!v!!')
        from chalicelib.new_class import EncoderCNN, DecoderRNN, Vocabulary
        print('!!w!!')
        from chalicelib.func import load_obj
        print('!!x!!')
        from torchvision import transforms
        print('!!y!!')
        import torch
        import torch.nn as nn
        import torchvision.models as models
        from torch.nn.utils.rnn import pack_padded_sequence
        print('!!z!!')

        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),  # Using ImageNet norms
                                 (0.229, 0.224,
                                  0.225))])  # https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2

        print('!!aa!!')

        list_vocab = load_obj('list_vocab')
        vocab = Vocabulary()

        # Add the token words first
        for i in list_vocab:
            vocab.add_word(i)
        #vocab = load_obj('vocab')
        # RNN models loading
        # rnn_decoder_5 = 'rnn-decoder-5.pt'
        # rnn_encoder_5 = 'rnn-encoder-5.pt'

        print('!!ab!!')

        encoder = EncoderCNN(embed_size)
        print('!!ac!!')
        decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
        print('!!ad!!')
        encoder.load_state_dict(torch.load(rnn_encoder_5_pt, map_location=torch.device('cpu')))
        print('!!ae!!')
        decoder.load_state_dict(torch.load(rnn_decoder_5_pt, map_location=torch.device('cpu')))
        print('!!af!!')

        # encoder.load_state_dict(torch.load(drive_path + rnn_encoder_5, map_location=torch.device('cpu')))
        # decoder.load_state_dict(torch.load(drive_path + rnn_decoder_5, map_location=torch.device('cpu')))
        rnn_model_5 = [encoder, decoder]
        print('!!ag!!')

        # FOR RNN
        encoder, decoder = rnn_model_5
        encoder.eval()
        with torch.no_grad():
            image = data_transform(im)
            # image=image.to(device)
            image = image.unsqueeze_(0)
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
            sampled_caption = sampled_caption[1:-1]
            sentence = ' '.join(sampled_caption)
            sentence = sentence + '.'
            ans = sentence.capitalize()

        reply_message(event, TextSendMessage(text=ans))

        # reply_message(event, TextSendMessage(text='OKです。'))

    except Exception as e:
        reply_message(event, TextSendMessage(text=f'エラーが発生しました:{e}'))

def reply_message(event, messages):
    linebot.reply_message(
        event.reply_token,
        messages=messages,
    )