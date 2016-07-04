# Neural Fuzzer

Neural-Fuzzer is an experimental fuzzer designed to use state-of-the-art Machine Learning to learn from a set of initial files. 
It works in two phases: **training** and **generation**.

* In training mode:  it uses [long-short term memory (LSTM)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn how sequences of bytes are structured. 

* In generation mode: it will automatically generate corrupted or unexpected files and it will try to crash a given program. 

Neural-Fuzzer is open-source (GPL3), powered by [keras](http://keras.io) and it is similar to [rnn-char](https://github.com/karpathy/char-rnn) and other techniques in sequence prediction.

## Requirements

* Python 2.7
* libhdf5-dev for saving and loading trained generators.
* [keras](http://keras.io) (automatically installed)
* [h5py](http://www.h5py.org/) (automatically installed)
* If you want to execute programs programs and triage crashes, you need to install gdb.
* If you are going to train your own file generators, you need a powerfull GPU.

## Installation

We need install the required libraries. For instance, in Debian/Ubuntu:

    # apt-get install python-numpy libhdf5-dev gdb

After that, we can start installing neural-fuzzer:

     $ git clone https://github.com/CIFASIS/neural-fuzzer/
     $ cd neural-fuzzer
     $ python setup.py install --user

## Example

### Generation of XML files

In order to generate XML, we can use one of the pre-trained XML generators:

    $ wget "https://github.com/CIFASIS/neural-fuzzer/releases/download/0.0/0-gen-xml.lstm"
    $ wget "https://github.com/CIFASIS/neural-fuzzer/releases/download/0.0/0-gen-xml.lstm.map"

(more generators are available [here](https://github.com/CIFASIS/neural-fuzzer/releases))

Then, we need a seed to start the generation. For instance, to use '>'

    $ mkdir seeds
    $ printf ">" > seeds/input.xml

Finally, we can start producing some random xmls using the generators:

    $  ./neural-fuzzer.py --max-gen-size 64 0-gen-xml.lstm seeds/
      Using Theano backend.
      Using ./gen-449983086021 to store the generated files
      Generating a batch of 8 file(s) of size 35 (temp: 0.5 )................................... 

The resulting files will be stored in a randomly named directory (e.g gen-449983086021). It is faster to generate files in a batch, instead of one by one (you can experiment with different batch sizes). In this case, one of the files we obtained is this one:

```xml
></p>
<p><termdef id='dt-encoding'>
```

An interesting parameter is the maximum size of the generated file. Another important parameter the temperature which takes a number
 in range (0, 1] (default = 0.5). As [karpathy explains](https://github.com/karpathy/char-rnn/blob/master/Readme.md), the temperature is dividing the predicted log probabilities before the Softmax, so lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes. 

## Generation and testing of XML parsing programs

Testing and triage of crashes using GDB can be performed using neural-fuzzer. For instance, if we want to test two XML parsing implementations libxml2 and expat:

    $  ./neural-fuzzer.py --max-gen-size 64 0-gen-xml.lstm seeds/ --cmd "/usr/bin/xmllint @@" "/usr/bin/xmlwf @@"

### Training

TODO
