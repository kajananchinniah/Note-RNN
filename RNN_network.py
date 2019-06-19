import torch
from torch import nn
import helper
import numpy as np
import torch.nn.functional as F

class noteRNN(nn.Module):
    def __init__(self, tokens, n_hidden, n_layers, dropout_prob = 0.5, lr = 0.01):
        super().__init__()
        
        self.dropout_prob = dropout_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        self.fc_hidden_size_1 = int((len(tokens) + n_hidden) / 2)
        self.notes = tokens
        self.int2note = dict(enumerate(self.notes))
        self.note2int = {note : i for i, note in self.int2note.items()}
        
        self.LSTM = nn.LSTM(len(self.notes), self.n_hidden, self.n_layers, dropout = self.dropout_prob, batch_first = True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(n_hidden, self.fc_hidden_size_1)
        self.fc2 = nn.Linear(self.fc_hidden_size_1, len(self.notes))
        self.logsoftmax = nn.LogSoftmax(dim = 1)
        
    def forward(self, x, hidden):
        out, hidden = self.LSTM(x, hidden)
        out = self.dropout(out)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out, hidden
    
    def init_hidden(self, batch_size, useGPU):
        weight = next(self.parameters()).data
        
        if (useGPU == True):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
                    
def train(model, train_data, valid_data,  file_save_to, epochs = 20, batch_size = 32, seq_length = 100, lr = 0.01, clip = 5, useGPU = False, save_every = 100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    
    if (useGPU == True):
        model.cuda()
    
    n_notes = len(model.notes)
    min_valid_loss = 9999999 #big random big number
    for e in range(1, epochs+1, 1):
        hid = model.init_hidden(batch_size, useGPU)
        train_loss_tot = 0
        train_count = 0
        for x, y in helper.getBatches(train_data, batch_size, seq_length):
            train_count = train_count + 1
            x = helper.oneHotEncode(x, n_notes)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if (useGPU == True):
                inputs = inputs.cuda()
                targets = targets.cuda()
                
            hid = tuple([each.data for each in hid])
            model.zero_grad()
            log_prob, hid = model(inputs, hid)
            loss = criterion(log_prob, targets.view(batch_size * seq_length).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss_tot = train_loss_tot + loss.item()
        
        
        else:
            valid_loss_tot = 0
            valid_count = 0
            valid_hid = model.init_hidden(batch_size, useGPU)
            with torch.no_grad():
                model.eval()
                for x, y in helper.getBatches(valid_data, batch_size, seq_length):
                    valid_count = valid_count + 1
                    x = helper.oneHotEncode(x, n_notes)
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                    
                    valid_hid = tuple([each.data for each in valid_hid])
                    if useGPU == True:
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    
                    log_prob, valid_hid = model(inputs, valid_hid)
                    valid_loss = criterion(log_prob, targets.view(batch_size * seq_length).long())
                    valid_loss_tot = valid_loss_tot + valid_loss.item()
                                       
                avg_train_loss = train_loss_tot / train_count 
                avg_valid_loss = valid_loss_tot / valid_count
                model.train()
                print("Epoch #: ", e)
                print("Train Loss : ", avg_train_loss)
                print("Valid Loss : ", avg_valid_loss)
                
                if (e % save_every == 0):
                    print("Saving model...")
                    f = str(e) + "_" + file_save_to
                    checkpoint = {'n_hidden' : model.n_hidden,
                                  'n_layers' : model.n_layers,
                                  'state_dict' : model.state_dict(),
                                  'tokens' : model.notes }
                    torch.save(checkpoint, f)
                    min_valid_loss = avg_valid_loss
                
def predict(model, note, hid = None, top_k = None, useGPU = False):
    x = np.array([[model.note2int[note]]])
    inputs = helper.oneHotEncode(x, len(model.notes))
    inputs = torch.from_numpy(inputs)
    
    if (useGPU == True):
        inputs = inputs.cuda()
       
    hid = tuple([each.data for each in hid])
    out, hid = model(inputs, hid)
    prob = F.log_softmax(out, dim = 1).data
    if (useGPU == True):
        prob = prob.cpu()
            
    if top_k == None:
        top_note = np.arange(len(model.notes))
    else:
        prob, top_note = prob.topk(top_k)
        top_note = top_note.numpy().squeeze()
    prob = prob.detach().numpy().squeeze()    
    note = np.random.choice(top_note, p = prob/prob.sum())
    
    return model.int2note[note], hid
    
def sample(model, size, start_notes, top_k, useGPU):
    if (useGPU == True):
        model.cuda()
    else:
        model.cpu()
        
    model.eval()
    
    notes = [start_notes]
    hid = model.init_hidden(1, useGPU)
    for n in [start_notes]:
        note, hid = predict(model, n, hid, top_k = top_k, useGPU = useGPU)            
    notes.append(note)
        
    for i in range(0, size, 1):
        note, hid = predict(model, notes[-1], hid, top_k = top_k, useGPU = useGPU)
        notes.append(note)
    return notes