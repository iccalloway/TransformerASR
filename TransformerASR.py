##Libraries

import os
os.chdir('C:/Users/Ian Calloway/Academics/Projects/ASR/')

import tensorflow as tf
import time
from numpy.random import random
from numpy import reshape, array, transpose, zeros, arange
from itertools import groupby


#import HelperFunctions
#from HelperFunctions import unpickle, empickle, prepare_array
from keras.layers import Conv2D, Input, Flatten,Dense,Reshape
from keras.models import Model
from keras.utils import normalize
from keras_transformer import get_model, get_model_ne

#Variables
features = 40
frames = 50
batches = 20
max_length = 20
offset = 2
dropout_rate = 0.1

def to_one_hot(a, embed_dim):
    b = zeros((a.size,embed_dim))
    b[arange(a.size),a]=1
    return b
    
##File Locations
segdict_location = "E:/Projects/English RNN Phonetic Classifier/segment_dict.pkl"
input_location = "./input3d.pickle"
output_location = "E:/Projects/English RNN Phonetic Classifier/TrainingOutput.pickle"
pad_list = ['<PAD>','<START>','<END>']

##Segment class setup
seg_dict = unpickle(segdict_location)
rev_seg = {item[1]:item[0] for item in seg_dict.items()}



combo = pad_list + list(seg_dict.keys())

full_dict = {combo[a]:a for a in range(len(combo))}
unique_segments = len(full_dict)

acoustics = unpickle(input_location)
acoustics = reshape(
    acoustics,
    (
        acoustics.shape[0],
        acoustics.shape[1],
        acoustics.shape[2],
        1
    )
)

out = unpickle(output_location)


o_array = array([full_dict[rev_seg[c]] for b in out for c in b]) ##Fix numbering for padding
output2d = prepare_array(reshape(o_array,(o_array.shape[0],1)), frames, 1)

out2d_flat = transpose(transpose(output2d)[0])
collapsed = [[x[0] for x in groupby(out2d_flat[b,:])] for b in range(out2d_flat.shape[0])]

decode_input = array(
    [
        [full_dict['<START>']]+
        a+
        [full_dict['<END>']]+
        [full_dict['<PAD>']]*(max_length - len(a)-2)
    for a in collapsed
    ]
)

decode_output = array([
    a+
    [full_dict['<END>']]+
    [full_dict['<PAD>']]*(max_length - len(a)-1)
    for a in collapsed])
decode_output = reshape(decode_output, (decode_output.shape[0],max_length,1))
embed_dim = 20
hidden_dim = 1000
adjusted_hd = frames*ceil(1000/frames)

'''
# Convolutional Block (Output 2D embedding tensor)
acous_input = Input(shape=(frames,features,1))
net = Conv2D(2,kernel_size=(5,5), strides=(2,2))(acous_input)
net = Conv2D(4,kernel_size=(5,5), strides=(2,2))(net)
net= Flatten()(net)
net = Dense(adjusted_hd)(net)
encode_input = Reshape((int(adjusted_hd/embed_dim),embed_dim))(net)
'''

#Shortened VGG Block
acous_input = Input(shape=(frames,features,1))
net = Conv2D(64,kernel_size=(3,3))(acous_input)
net = Conv2D(64,kernel_size=(3,3))(net)
net = MaxPool2D(pool_size=(2,2),strides=(2,2))(net)
net = Conv2D(128,kernel_size=(3,3))(net)
net = Conv2D(128,kernel_size=(3,3))(net)
net = MaxPool2D(pool_size=(2,2),strides=(2,2))(net)
net = Conv2D(256,kernel_size=(3,3))(net)
net = Conv2D(256,kernel_size=(3,3))(net)
net = MaxPool2D(pool_size=(2,2),strides=(2,2))(net)
net= Flatten()(net)
net = Dense(adjusted_hd)(net)
encode_input = Reshape((int(adjusted_hd/embed_dim),embed_dim))(net)


# Transformer Block (Output 1D segment prediction tensor)
model = get_model_ne(
    token_num=[280, unique_segments],
    embed_dim=embed_dim,
    encode_input=encode_input,
    encode_start=acous_input,
    encoder_num=2,
    decoder_num=2,
    head_num=4,
    hidden_dim=100,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    use_same_embed=False,
    embed_trainable=[True, True],
    embed_weights=[random((280,20)),random((77,20))]
)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x=[acoustics,decode_input],
    y=decode_output,
    epochs=5,
    validation_split=0.2
)

decode(
    model,
    acoustics[100,:,1:2],
    full_dict['<START>'],
    full_dict['<END>'],
    full_dict['<PAD>'],
)