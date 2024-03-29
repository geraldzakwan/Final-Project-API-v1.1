from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding, Bidirectional, Concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import os
import sys
import zipfile

import attention_lstm

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

np.random.seed(42)

# BATCH_SIZE = int(os.environ['BATCH_SIZE'])
BATCH_SIZE = 64
NUM_EPOCHS = int(sys.argv[3])
GLOVE_EMBEDDING_SIZE = int(os.environ['GLOVE_EMBEDDING_SIZE'])
HIDDEN_UNITS = int(os.environ['HIDDEN_UNITS'])
MAX_INPUT_SEQ_LENGTH = int(os.environ['MAX_INPUT_SEQ_LENGTH'])
MAX_TARGET_SEQ_LENGTH = int(os.environ['MAX_TARGET_SEQ_LENGTH'])
MAX_VOCAB_SIZE = int(os.environ['MAX_VOCAB_SIZE'])
DATA_SET_NAME = 'gunthercox'
DATA_DIR_PATH = 'data/gunthercox'

GLOVE_MODEL = "very_large_data/glove.6B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'

# WEIGHT_FILE_PATH = 'models/' + DATA_SET_NAME + '/word-glove-weights.h5'
MODEL_NAME = sys.argv[2]
WEIGHT_FILE_PATH = 'models/' + DATA_SET_NAME + '/' + MODEL_NAME + '.h5'


def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove():
    if not os.path.exists(GLOVE_MODEL):

        glove_zip = 'very_large_data/glove.6B.zip'

        if not os.path.exists('very_large_data'):
            os.makedirs('very_large_data')

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall('very_large_data')
        zip_ref.close()


def load_glove():
    download_glove()
    _word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em

word2em = load_glove()

target_counter = Counter()

input_texts = []
target_texts = []

for file in os.listdir(DATA_DIR_PATH):
    filepath = os.path.join(DATA_DIR_PATH, file)
    if os.path.isfile(filepath):
        print('processing file: ', file)
        lines = open(filepath, 'rt', encoding='utf8').read().split('\n')
        prev_words = []
        for line in lines:

            if line.startswith('- - '):
                prev_words = []

            if line.startswith('- - ') or line.startswith('  - '):
                line = line.replace('- - ', '')
                line = line.replace('  - ', '')
                next_words = [w.lower() for w in nltk.word_tokenize(line)]
                next_words = [w for w in next_words if in_white_list(w)]
                if len(next_words) > MAX_TARGET_SEQ_LENGTH:
                    next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

                if len(prev_words) > 0:
                    input_texts.append(prev_words)

                    target_words = next_words[:]
                    target_words.insert(0, 'start')
                    target_words.append('end')
                    for w in target_words:
                        target_counter[w] += 1
                    target_texts.append(target_words)

                prev_words = next_words

for idx, (input_words, target_words) in enumerate(zip(input_texts, target_texts)):
    if idx > 10:
        break
    print([input_words, target_words])

target_word2idx = dict()
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1

if 'unknown' not in target_word2idx:
    target_word2idx['unknown'] = 0

target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_decoder_tokens = len(target_idx2word)

np.save('models/' + DATA_SET_NAME + '/word-glove-target-word2idx.npy', target_word2idx)
np.save('models/' + DATA_SET_NAME + '/word-glove-target-idx2word.npy', target_idx2word)

input_texts_word2em = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(input_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        if w in word2em:
            emb = word2em[w]
        encoder_input_wids.append(emb)

    input_texts_word2em.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

print(context)
np.save('models/' + DATA_SET_NAME + '/word-glove-context.npy', context)


def generate_batch(input_word2em_data, output_text_data):
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, GLOVE_EMBEDDING_SIZE))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = target_word2idx['unknown']  # default unknown
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    if w in word2em:
                        decoder_input_data_batch[lineIdx, idx, :] = word2em[w]
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


if('attention' in sys.argv[1]):
    # THIS IS STILL RANDOM IDEA
    # encoder_inputs = Input(shape=(None, MAX_INPUT_SEQ_LENGTH, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
    encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
else:
    encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')

if(sys.argv[1] == 'bidirectional'):
    print('TRAINING ON BIDIRECTIONAL')

    encoder_lstm = Bidirectional(LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm'))
    encoder_outputs, encoder_state_forward_h, encoder_state_forward_c, encoder_state_backward_h, encoder_state_backward_c = encoder_lstm(encoder_inputs)

    # IF BIDIRECTIONAL, NEEDS TO CONCATENATE FORWARD AND BACKWARD STATE
    encoder_state_h = Concatenate()([encoder_state_forward_h, encoder_state_backward_h])
    encoder_state_c = Concatenate()([encoder_state_forward_c, encoder_state_backward_c])
else:
    encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')

    if('attention' in sys.argv[1]):
        # THIS IS STILL RANDOM IDEA TO IGNORE THE 2ND DIMENSION
        # encoder_outputs, _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
    else:
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)

encoder_states = [encoder_state_h, encoder_state_c]

if(sys.argv[1] == 'bidirectional'):
    decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
    decoder_lstm = LSTM(units=HIDDEN_UNITS * 2, return_state=True, return_sequences=True, name='decoder_lstm')
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)
else:
    if('attention' in sys.argv[1]):
        # HERE, THE GLOVE EMBEDDING SIZE ACTS AS THE INPUT DIMENSION
        # IF USING ATTENTION, WE NEED TO SET SHAPE WITH TIME STEPS, NOT WITH NONE
        # THIS INPUT WILL BE USED WHEN BUILDING ENCODER OUTPUTS

        # decoder_inputs = Input(shape=(None, attention_lstm.TIME_STEPS, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        # decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        # decoder_inputs = Input(shape=(MAX_TARGET_SEQ_LENGTH + 2, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        decoder_inputs = Input(shape=(decoder_max_seq_length, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')

        if(sys.argv[1] == 'attention_before'):
            attention_mul = attention_lstm.attention_3d_block(decoder_inputs, decoder_max_seq_length)
    else:
        decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')

    # PAY ATTENTION THAT DECODER AND ENCODER STATE MUST ALWAYS HAVE THE SAME DIMENSION
    # IN THIS CASE, WE USE 2D
    decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')

    if('attention' in sys.argv[1]):
        # REMOVE ENCODER AS INITIAL STATE FOR ATTENTION
        # decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs)
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)
    else:
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                     initial_state=encoder_states)

if(sys.argv[1] == 'attention_after'):
    attention_mul = attention_lstm.attention_3d_block(decoder_outputs, decoder_max_seq_length)
    # SOMEHOW THIS FLATTEN FUNCTION CAUSE THE PROBLEM
    # attention_mul = Flatten()(attention_mul)

decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')

if(sys.argv[1] == 'attention_after' or sys.argv[1] == 'attention_before'):
    decoder_outputs = decoder_dense(attention_mul)
else:
    decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

json = model.to_json()
open('models/' + DATA_SET_NAME + '/word-glove-architecture.json', 'w').write(json)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, target_texts, test_size=0.2, random_state=42)

print(len(Xtrain))
print(len(Xtest))

train_gen = generate_batch(Xtrain, Ytrain)
test_gen = generate_batch(Xtest, Ytest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

# checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

# CALLBACK TO STOP IF THERE IS NO IMPROVEMENTS AND TO SAVE CHECKPOINTS
callbacks = [
    EarlyStopping(monitor='val_loss', min_delta=0, patience=int(os.environ['PATIENCE']), verbose=0, mode='auto'),
    ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
]

model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=callbacks)

model.save_weights(WEIGHT_FILE_PATH)
