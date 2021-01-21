from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, Dense, Flatten, LSTM
from keras import Input
from keras import layers
from keras.callbacks import ModelCheckpoint

import numpy as np
import h5py
import datetime

filter = '!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n0123456789'


def writeDict(filename, dict):
    with h5py.File(filename, 'w') as h5file:
        for key, item in dict.items():
            # print(key,item)
            h5file["/" + str(key)] = item


def readDict(filename):
    dict = {}
    with h5py.File(filename, 'r') as h5file:
        for key, item in h5file["/"].items():
            dict[key] = item.value
    return dict


def createWordIndex(inputfile, outputIndexFile):
    titles, texts, _ = readFile(inputfile)
    tokenizer = Tokenizer(num_words=10000, filters=filter)
    tokenizer.fit_on_texts(titles + texts)
    writeDict(outputIndexFile, tokenizer.word_index)


def convertWordSequenceToIndex(ws, word_index, maxLen):
    vect = np.full((len(ws), maxLen), 0, dtype='int32')
    for i in range(0, len(ws)):
        for j in range(0, len(ws[i])):
            if j < maxLen:
                if word_index.get(ws[i][j]) > 0:
                    vect[i][j] = word_index.get(ws[i][j])
    return vect


def readFile(inputfile):
    titles = []
    texts = []
    labels = []
    with open(inputfile, 'r') as inputf:
        for line in inputf:
            fields = line.rstrip().split('\t')
            if len(fields) == 3:
                titles.append(fields[0])
                texts.append(fields[1])
                labels.append(fields[2])
            else:
                print(line)
    return titles, texts, labels


def get_model(maxWords, maxLenTitle, maxLenDesp, numOfCategory, dims=32):
    input_tensor1 = Input(shape=(maxLenTitle,))
    x1 = Embedding(maxWords, dims)(input_tensor1)
    x1 = LSTM(dims)(x1)

    input_tensor2 = Input(shape=(maxLenDesp,))
    x2 = Embedding(maxWords, dims)(input_tensor2)
    x2 = LSTM(dims)(x2)

    merge_layer = layers.concatenate([x1, x2], axis=-1, name='feature_vector')
    output_tensor = Dense(numOfCategory, activation='softmax')(merge_layer)

    model = Model([input_tensor1, input_tensor2], output_tensor)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    return model


def data_generator(x1, x2, y, batch_size, shuffle=True, name=''):
    batches = len(y) // batch_size
    while 1:
        if shuffle:
            idx = np.random.permutation(len(y))
            x1 = x1[idx]
            x2 = x2[idx]
            y = y[idx]

        for i in range(0, batches):
            yield [x1[i * batch_size: (i + 1) * batches], x1[i * batch_size: (i + 1) * batches]], y[i * batch_size: (
                                                                                                                                i + 1) * batches]


########################### main function ########################

title_len = 20
body_len = 100

if __name__ == '__main__':
    # createWordIndex('macys.tsv', 'macy_products.index')
    wordIdx = readDict('macy_products.index')

    model = get_model(1000, title_len, body_len, 30)
    model.summary()

    model_fn = "multi-squeeze_%s-{epoch:02d}-{val_acc:0.2f}.hdf5" % (datetime.datetime.now().strftime("%m%d%H"))

    ckpt = ModelCheckpoint(model_fn,
                           monitor='val_acc',
                           save_best_only=True,
                           mode='max')

    train_titles, train_body, train_y = readFile('train.set')
    train_x1 = convertWordSequenceToIndex(train_titles, wordIdx, title_len)
    train_x2 = convertWordSequenceToIndex(train_desp, wordIdx, body_len)

    test_titles, test_body, test_y = readFile('test.set')
    test_x1 = convertWordSequenceToIndex(test_titles, wordIdx, title_len)
    test_x2 = convertWordSequenceToIndex(test_body, wordIdx, body_len)

'''
    history = model.fit_generator(data_generator(train_x1, 
                                       train_x2, 
                                       train_y, 
                                        batch_size=128), 
                        steps_per_epoch=1500, 
                        epochs=60,
                        validation_data=data_generator(test_x1, 
                                                        test_x2, 
                                                        test_y, 
                                                        batch_size=128),
                        validation_steps=100,
                        callbacks=[ckpt]
                        )
    print(history['acc'])
'''
