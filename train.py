from pathlib import Path
import numpy as np
import helper
import torch
import RNN_network

#Constants
data_dir = 'All/Classical/'
useGPU =  torch.cuda.is_available()
n_hidden = 512
n_layers = 3
seq_length = 100
batch_size = 128
valid_percent = 0.3
n_epochs = 200
lr = 0.1 #mostly guess and checked 
file_save_to = 'All_NoteRNN.pt'

p = Path(data_dir)
files = p.glob("*.mid")
notes = helper.processData(files)

pitchnames = tuple(set(notes))
int2note = dict(enumerate(pitchnames))
note2int = {note : i for i, note in int2note.items()}
encoded = np.array([note2int[note] for note in notes])
valid_index = int(len(encoded) * (1 - valid_percent))
encoded_train, encoded_valid = encoded[:valid_index], encoded[valid_index:]
model = RNN_network.noteRNN(pitchnames, n_hidden, n_layers, lr = lr)
print(model)
RNN_network.train(model, encoded_train, encoded_valid, file_save_to, epochs = n_epochs, batch_size = batch_size, seq_length = seq_length, lr = lr, useGPU = useGPU, saveMinTrainLoss = True)