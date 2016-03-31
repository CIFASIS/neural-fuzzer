#!/usr/bin/env python

''' xxx '''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import argparse
import numpy as np
import random
import os
import sys

from triage import triage

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def read_seeds(seeds, nsamples):

  all_files = []

  for x,y,files in os.walk(seeds):
    #print(files, x, y)
    for f in files:
      #all_files.append(x+"/".join(y)+"/"+f)
      all_files.append(x+"/"+f)


  random.shuffle(all_files)

  all_files = all_files[0:nsamples]
  #print(all_files)
  seeds_text = ""

  for filename in all_files:
    seeds_text = seeds_text + (open(filename).read())
    #text = text + "\n\n\n" + x #filter(lambda y: y in string.printable, x).lower()

  return seeds_text

if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("seeds", help="", type=str, default=None)
    parser.add_argument("cmd", help="", type=str, default=None)
    #parser.add_argument("-d", help="", type=int, default=5)
    #parser.add_argument("-p", help="", action="store_true", default=False)

    parser.add_argument("--model", type=str,
                        help="",
                        action="store", default=None)

    parser.add_argument("--gen",
                        help="Test a model using infile (recall only)",
                        action="store_true", default=False)

    parser.add_argument("--max-gen-size", type=int,
                        help="",
                        action="store", default=100)

    parser.add_argument("--n-samples", type=int,
                        help="",
                        action="store", default=10)

    options = parser.parse_args()
    seeds = options.seeds
    cmd = options.cmd

    gen_mode = options.gen
    nsamples = options.n_samples
    maxgenlen = options.max_gen_size

    #depth = options.d
    #prune = options.p

    #assert(0)

    text = read_seeds(seeds, nsamples)

    print('corpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    print(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 20
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    #assert(0)

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1


    # build the model: 2 stacked LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #assert(0)

    # train the model, output generated text after each iteration
    for iteration in range(1, 50):
        #print()
        #print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=10)

        start_index = random.randint(0, len(text) - maxlen - 1)


        for diversity in [0.2, 0.5, 1.0, 1.2]:
            #print()
            filename = "test/gen-"+str(iteration)+"-"+str(diversity)
            f = open(filename, "w+")

            #print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            #print('----- Generating with seed: "' + sentence + '"')
            #print('saving at '+ filename)

            f.write(generated)
            gensize = random.randint(maxgenlen / 2, maxgenlen)


            for i in range(gensize):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                f.write(next_char)
                #f.flush()
            #print()
            f.close()

        results = triage(cmd, "test")
        for (k,v) in results.items():
          if k <> "":
            print(k,v)
            assert(0)

    print(indices_char)
    model.save_weights('test.h5', overwrite=True)
