# Note-RNN

Note RNN is my attempt at creating a RNN to generate music. During my initial attempt, I had a lot of trouble training, so I tried a couple of different things to get my model to actually train. 

RNN_network.py contains the class definition of the RNN, a function to train the RNN, a function to predict the next note in the sequence, and a function to generate x amounts of notes.

train.py is a script that trains the model

test.py is a script that gets the output of the neural network from a random starting note, and then generates a midi file using it

helper.py contains helper functions used (one hot encoding, getting the batches, and converting and saving the file as a midi 

The folder model files contains the models I obtained from training for epochs 100 - 400, and a sample midi output for each model saved. 

Some changes I made from my last iteration that seemed to improve my training & results:
* Increase my fully connected layers from 1 to 2 
* Changed my optimizer from SGD to Adam. Adam seemed to have given me the best results compared to other optimization algorithms I used (I tried Adadelta, rmsprop and Adagrad)
* Used more data (my chrono model now uses the soundtrack from chrono trigger and chrono cross, as for my all model, it uses both chrono games soundtracks, the soundtrack from the legend of zelda ocarina of time, and pieces composed by some classical musicians.  The all data was mostly just random things).
* Decreased number of layers in lstm
* allow for more variance in note chosen (I select from the top 10 now instead of the top 2)

These changes led to overfitting, and poor performance on my validation set however. Sometimes the midi sounds like a weird combination of a few songs in my opinion. 

For preprocessing & midi generation, I used this:
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

For setting up my RNN, using the results and setting up some other functions (i.e. getting batches), I used this as reference:
https://github.com/udacity/deep-learning-v2-pytorch/tree/master/recurrent-neural-networks/char-rnn
