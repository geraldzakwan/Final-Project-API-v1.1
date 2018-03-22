from cornell_char_seq2seq_predict import CornellCharChatBot
from cornell_word_seq2seq_predict import CornellWordChatBot
from cornell_word_seq2seq_glove_predict import CornellWordGloveChatBot
from gunthercox_word_seq2seq_predict import GunthercoxWordChatBot
from gunthercox_word_seq2seq_glove_predict import GunthercoxWordGloveChatBot
from gunthercox_char_seq2seq_predict import GunthercoxCharChatBot

import os
import sys
import nltk

from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort, session
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, PostbackEvent, TextMessage, TextSendMessage, ImageMessage, ImageSendMessage, ButtonsTemplate,
    URITemplateAction, TemplateSendMessage
)

from urllib import parse, request as req
import jwt, json

load_dotenv(find_dotenv(), override=True)
app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

cornell_char_chat_bot = CornellCharChatBot()
cornell_word_chat_bot = CornellWordChatBot()
gunthercox_char_chat_bot = GunthercoxCharChatBot()
gunthercox_word_chat_bot = GunthercoxWordChatBot()
gunthercox_word_glove_chat_bot = GunthercoxWordGloveChatBot()
cornell_word_glove_chat_bot = CornellWordGloveChatBot()

cornell_char_chat_bot_conversations = []
cornell_word_chat_bot_conversations = []
cornell_word_glove_chat_bot_conversations = []
gunthercox_char_chat_bot_conversations = []
gunthercox_word_chat_bot_conversations = []
gunthercox_word_glove_chat_bot_conversations = []

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)

if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

chatbot_state = {}
chatbot_state['dataset'] = 'cornell'

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    in_text = event.message.text.lower()
    user_id = event.source.user_id

    if('set dataset to' in in_text):
        tokens = nltk.tokenize(in_text)
        dataset_name = tokens[len(tokens) - 1]
        chatbot_state['dataset'] = dataset_name
        res_text = 'You are now talking with ' + dataset_name + ' chatbot'
    else:
        response = redirect('chatbot_reply?sentence=' + in_text + '&level=word-glove&dialogs=' + chatbot_state['dataset'])
        print(response)
        res_text = 'test'

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res_text)
    )

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return 'About Us'

@app.route('/cornell_char_reply', methods=['POST', 'GET'])
def cornell_char_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_char_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_char_chat_bot.reply(sent)
            cornell_char_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_char_reply.html', conversations=cornell_char_chat_bot_conversations)

@app.route('/cornell_word_reply', methods=['POST', 'GET'])
def cornell_word_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_word_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_word_chat_bot.reply(sent)
            cornell_word_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_word_reply.html', conversations=cornell_word_chat_bot_conversations)

@app.route('/gunthercox_char_reply', methods=['POST', 'GET'])
def gunthercox_char_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_char_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_char_chat_bot.reply(sent)
            gunthercox_char_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_char_reply.html', conversations=gunthercox_char_chat_bot_conversations)

@app.route('/gunthercox_word_reply', methods=['POST', 'GET'])
def gunthercox_word_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_word_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_word_chat_bot.reply(sent)
            gunthercox_word_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_word_reply.html', conversations=gunthercox_word_chat_bot_conversations)

@app.route('/gunthercox_word_glove_reply', methods=['POST', 'GET'])
def gunthercox_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            gunthercox_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = gunthercox_word_glove_chat_bot.reply(sent)
            gunthercox_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('gunthercox_word_glove_reply.html', conversations=gunthercox_word_glove_chat_bot_conversations)

@app.route('/cornell_word_glove_reply', methods=['POST', 'GET'])
def cornell_word_glove_reply():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            cornell_word_glove_chat_bot_conversations.append('YOU: ' + sent)
            reply = cornell_word_glove_chat_bot.reply(sent)
            cornell_word_glove_chat_bot_conversations.append('BOT: ' + reply)
    return render_template('cornell_word_glove_reply.html', conversations=cornell_word_glove_chat_bot_conversations)

@app.route('/chatbot_reply', methods=['POST', 'GET'])
def chatbot_reply():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'level' not in request.json or 'dialogs' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        level = request.json['level']
        dialogs = request.json['dialogs']
    else:
        sentence = request.args.get('sentence')
        level = request.args.get('level')
        dialogs = request.args.get('dialogs')

    target_text = sentence
    if level == 'char' and dialogs == 'cornell':
        target_text = cornell_char_chat_bot.reply(sentence)
    elif level == 'word' and dialogs == 'cornell':
        target_text = cornell_word_chat_bot.reply(sentence)
    elif level == 'word-glove' and dialogs == 'cornell':
        target_text = cornell_word_glove_chat_bot.reply(sentence)
    elif level == 'char' and dialogs == 'gunthercox':
        target_text = gunthercox_char_chat_bot.reply(sentence)
    elif level == 'word' and dialogs == 'gunthercox':
        target_text = gunthercox_word_chat_bot.reply(sentence)
    elif level == 'word-glove' and dialogs == 'gunthercox':
        target_text = gunthercox_word_glove_chat_bot.reply(sentence)
    return jsonify({
        'sentence': sentence,
        'reply': target_text,
        'dialogs': dialogs,
        'level': level
    })

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

def main():
    # cornell_char_chat_bot.test_run()
    # cornell_word_chat_bot.test_run()
    # cornell_word_glove_chat_bot.test_run()
    # gunthercox_char_chat_bot.test_run()
    # gunthercox_word_chat_bot.test_run()
    # gunthercox_word_glove_chat_bot.test_run()

    app.secret_key = os.urandom(12)
    try:
        port = int(os.environ['PORT'])
    except KeyError:
        print('Specify PORT as environment variable.')
        sys.exit(1)
    except TypeError:
        print('PORT must be an integer.')
        sys.exit(1)

    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    main()
