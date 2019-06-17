from music21 import converter, instrument, note, chord, stream
import numpy as np

def processData(data):
    notes = []
    for file in data:
        midi = converter.parse(file)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def oneHotEncode(arr, n_labels):
    return np.eye(n_labels, dtype = np.float32)[arr,:]

def getBatches(arr, batch_size, seq_length):
    n_batches = len(arr) // (batch_size * seq_length)
    if n_batches == 0:
        print('Error: insufficient data!')
        return
    arr = arr[:n_batches * batch_size * seq_length] #remove elements we can't create full elements with
    arr = arr.reshape((batch_size, -1)) #first dimension is batch size
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length] #Features
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length] #shift targets by one
        except IndexError: #Wrap it around instead
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        
  
def convertToInstrument(file_save_to, instrument_str = 'piano'):
    if instrument_str.lower() == 'piano':
        chosen_instrument = instrument.Piano()
    elif instrument_str.lower() == 'violin':
        chosen_instrument = instrument.Violin()
    elif instrument_str.lower() == 'guitar':
        chosen_instrument =  instrument.AcousticGuitar()
    else:
        print('Unrecognized instrument chosen!')
        return None

    s = converter.parse(file_save_to)
    for i in s.recurse():
        if 'Instrument' in i.classes: # Changing all instruments to the type chosen
            i.activeSite.replace(i, chosen_instrument)
    s.write('midi', file_save_to)


def convertAndSaveMidi(predictions, file_save_to, instrument_str = 'piano'):
    output_notes = []
    offset = 0
    
    for pattern in predictions:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset = offset + 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp = file_save_to)
    convertToInstrument(file_save_to, instrument_str)