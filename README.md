# Note-RNN

Currently, this doesn't work very well. I think the problem lies with my network architecture as it's far too simple, but my model also doesn't seem to be training well either (converges very quickly).

My future plans are:
1) Try to purposely get my network to overfit. Currently, I'd say that my network isn't training well enough from the data.
2) Research other, more complex network architectures to try and get better results (e.g. encoder decoder)

RNN_network.py contains the class definition of the RNN, a function to train the RNN, a function to predict the next note in the sequence, and a function to generate x amounts of notes.

train.py is a script that trains the model

test.py is a script that gets the output of the neural network from a random starting note, and then generates a midi file using it

helper.py contains helper functions used (one hot encoding, getting the batches, and converting and saving the file as a midi 

For preprocessing, I used this as reference:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

For setting up my RNN, using the results and setting up some other functions (i.e. getting batches), I used this as reference:
https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks/char-rnn
