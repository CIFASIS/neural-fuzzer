#!/usr/bin/env python

''' TODO '''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
import argparse
import numpy as np
import random
import pickle
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

def recall(model, char_indices, indices_char, data, testdirs, filename, maxlen, maxgenlen):

    f = open(filename, "w+")
    f.write(data)

    #if len(data) < maxlen:
    #  data = "".join(map(chr, list(np.random.random_integers(0,255,maxlen-len(data)))  )) + data
    #  data = "".join(map(chr, [0]*(maxlen-len(data))  )) + data

    #print ("Using",data,"as input.")

    generated = ''
    sentence = data
    generated += sentence

    gensize = random.randint(maxgenlen / 2, maxgenlen)
    model.reset_states()

    for i in range(gensize):
        x = np.zeros((1, maxlen, len(char_indices)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        f.write(next_char)
        #f.flush()

    f.close()

def define_model(input_dim, output_dim):

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("model", help="", type=str, default=None)
    parser.add_argument("seeds", help="", type=str, default=None)
    parser.add_argument("cmd", help="", type=str, default=None)
    #parser.add_argument("-d", help="", type=int, default=5)
    #parser.add_argument("-p", help="", action="store_true", default=False)

    #parser.add_argument("--model", type=str,
    #                    help="",
    #                    action="store", default=None)

    parser.add_argument("--gen",
                        help="Test a model using infile (recall only)",
                        action="store_true", default=False)

    parser.add_argument("--max-gen-size", type=int,
                        help="",
                        action="store", default=100)

    parser.add_argument("--n-samples", type=int,
                        help="",
                        action="store", default=sys.maxsize)


    parser.add_argument("--start-index", type=int,
                        help="",
                        action="store", default=None)



    options = parser.parse_args()
    file_model = options.model
    seeds = options.seeds
    cmd = options.cmd

    gen_mode = options.gen
    nsamples = options.n_samples
    maxgenlen = options.max_gen_size
    fixed_start_index = options.start_index

    #depth = options.d
    #prune = options.p

    #assert(0)

    text = read_seeds(seeds, nsamples)
    maxlen = 20

    if gen_mode:
        (char_indices, indices_char) = pickle.load(open(file_model+".map","r"))
        model = define_model((maxlen, len(char_indices)), len(char_indices))
        model.load_weights(file_model)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        for iteration in range(0,10):
            for diversity in [x / 100.0 for x in range(1,151,10)]:
                sys.stdout.write('.')
                sys.stdout.flush()
                #print('.', end="")
                filename = "test/gen-"+str(iteration)+"-"+str(diversity)
                recall(model, char_indices, indices_char, text, "test", filename, maxlen, maxgenlen)

        print(triage(cmd, "test"))
        sys.exit(0)

    print('corpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    print(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
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
    model = define_model((maxlen, len(chars)), len(chars))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # train the model, output generated text after each iteration
    for iteration in range(0, 1):
        #print()
        #print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=128, nb_epoch=5)

        if  fixed_start_index:
            start_index = fixed_start_index
        else:
            start_index = random.randint(0, len(text) - maxlen - 1)

        for diversity in [x / 10.0 for x in range(1,10)]:

            #print()
            filename = "test/gen-"+str(iteration)+"-"+str(diversity)
            recall(model, char_indices, indices_char, text[start_index: start_index + maxlen], "test", filename, maxlen, maxgenlen)

        #cmd = ""
        #os.system(cmd)

        results = triage(cmd, "test")
        for (k,v) in results.items():
          if k <> "":
            print(k,v)
            assert(0)

    #print(indices_char)
    pickle.dump((char_indices, indices_char), open(file_model+".map","w+"))
    model.save_weights(file_model, overwrite=True)
