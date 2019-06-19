import torch
import random

import RNN_network
import helper

file_load_from = '400_all_NoteRNN.pt'
top_k = 10
useGPU = torch.cuda.is_available()
midi_file = '400_all.mid'

#If no GPU is avaliable, load using the CPU 
if useGPU == False:
    checkpoint = torch.load(file_load_from, map_location = 'cpu')
else:
    checkpoint = torch.load(file_load_from)
    
model = RNN_network.noteRNN(checkpoint['tokens'], n_hidden = checkpoint['n_hidden'], n_layers = checkpoint['n_layers'])    
model.load_state_dict(checkpoint['state_dict'])

if useGPU == True:
    model = model.cuda()
    
random_int = random.randint(0, len(model.notes) - 1) # getting a random integer from 0 to number of indexes
output = RNN_network.sample(model, 500, model.int2note[random_int], top_k, useGPU)
helper.convertAndSaveMidi(output, midi_file)