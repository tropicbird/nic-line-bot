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

#---------original
# import pandas as pd
#
# from chalicelib.character import character
# df = pd.read_csv(f"chalicelib/{character}_other_fixed.csv")
# df_nage = pd.read_csv(f"chalicelib/{character}_nage_fixed.csv")
#
# comment = "入力に誤りがあります。\n" \
#       "ID番号 or コマンド or 技名を入力してください。\n" \
#       "(例：121 or LPRP or 風神拳）\n" \
#       "\n" \
#       "「一覧」と入力するとID番号が確認できます。\n" \
#       "\n" \
#       "※表記ルール\n" \
#       "右手：RP、左手：LP、右足：RK、左足：LK、" \
#       "両手：WP、両足：WK、右手と右足：WR、右手と右足：WL、" \
#       "右手と左足：RPK、左手と右足：LPK\n" \
#       "↙：1、↓：2、↘：3、←：4、→：6、↖：7、↑：8、↗：9\n" \
#       "長押し：'、ニュートラル：☆、スライド入力：【】"
#---------original

logger = logging.getLogger()
app = Chalice(app_name='line-api')
handler = WebhookHandler(os.environ.get('LINE_CHANNEL_SECRET'))
linebot = LineBotApi(os.environ.get('LINE_CHANNEL_ACCESS_TOKEN'))

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
    if not _valid_reply_token(event):
        return

    ans = "画像を送信してね。"
    reply_message(event, TextSendMessage(text=ans))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    print("handle_image:", event)

    try:
        reply_message(event, TextSendMessage(text='OKです。'))

    except Exception as e:
        reply_message(event, TextSendMessage(text='エラーが発生しました'))

def reply_message(event, messages):
    linebot.reply_message(
        event.reply_token,
        messages=messages,
    )