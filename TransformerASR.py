# TODO
# Improve class balance
# Libraries
from numpy.random import random
from numpy import reshape, array, transpose, zeros, arange, unique
from itertools import groupby
from sklearn.utils import class_weight
from HelperFunctions import unpickle, empickle, prepare_array
from collections import Counter
import tensorflow as tf


# Keras-related functions
from keras import metrics
from keras.layers import Conv2D, Input, Flatten, Dense, Reshape
from keras.models import Model
from keras.utils import normalize

# Custom Alterations to keras_transformer
from transformer_custom import get_model_ne, decode_ne

# Variables
features = 40
frames = 50
batches = 20
max_length = 20
offset = 2
dropout_rate = 0.1

# Tokens to be excluded from analysis (Most are non-existent)
exclude = ['<PAD>','a','x','e','q','h','i','id','no','hhn','an','j']

# Segment classes that need to be split into two
transforms = {
    'ihn':['ih', 'n'],
    'own':['ow', 'n'],
    'ayn':['ay', 'n'],
    'aen':['ae', 'n'],
    'aan':['aa', 'n'],
    'ahn':['ah', 'n'],
    'iyn':['iy','n'],
    'uhn':['uh','n'],
    'awn':['aw', 'n'],
    'eyn':['ey', 'n'],
    'aon':['ao', 'n'],
    'uwn':['uw', 'n'],
    'oyn':['oy', 'n'],
    'ern':['er','n'],
    'ehn':['eh','n']
}

def remove_by_indices(iter, idxs):
    return [e for i, e in enumerate(iter) if i not in idxs]

def renumber(d,uniques):
    return {uniques[a]:a  for a in range(len(uniques))}



# File Locations
segdict_location = "E:/Projects/English RNN Phonetic Classifier/segment_dict.pkl"
input_location = "./input3d.pickle"
output_location = "E:/Projects/English RNN Phonetic Classifier/TrainingOutput.pickle"
pad_list = ['<PAD>', '<START>', '<END>']

# Segment class setup
seg_dict = unpickle(segdict_location)
rev_seg = {item[1]: item[0] for item in seg_dict.items()}

full_keys = pad_list + list(seg_dict.keys())
full_dict = {full_keys[a]: a for a in range(len(full_keys))}
rev_full = {item[1]: item[0] for item in full_dict.items()}


# Encoder input setup
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

# Decoder input/output setup
out = unpickle(output_location)
o_array = array([full_dict[rev_seg[c]] for b in out for c in b])  ##Fix numbering for padding

output2d = prepare_array(reshape(o_array, (o_array.shape[0], 1)), frames, 1)

out2d_flat = transpose(transpose(output2d)[0])
collapsed = [[x[0] for x in groupby(out2d_flat[b, :])] for b in range(out2d_flat.shape[0])]

# Determine which frames to exclude
exclude_vals = [full_dict[a] for a in exclude]
exclude_indices =where(array([a if any(b in collapsed[a] for b in exclude_vals) else None for a in range(len(collapsed))])!=None)
excluded = remove_by_indices(transform_vals, exclude_indices[0].tolist())

# Determine which segment representations to simplify
convert_dict = dict([(full_dict[a[0]],[full_dict[b] for b in a[1]]) for a in transforms.items()])
test=[[convert_dict[b] if b in convert_dict else [b] for b in a] for a in excluded]
transform_vals = [[item for sublist in a for item in sublist] for a in test]


flatten = [item for sublist in transform_vals for item in sublist]
flat_count = Counter(flatten)
new_uniques = pad_list+[rev_full[a]  for a in flat_count.keys()]
final_dict = renumber(full_dict, new_uniques)
rev_final = {item[1]: item[0] for item in final_dict.items()}

unique_segments = len(final_dict)

collapsed = [[final_dict[rev_full[b]] for b in a] for a in transform_vals]




decode_input = array(
    [
        [full_dict['<START>']] +
        a +
        [full_dict['<END>']] +
        [full_dict['<PAD>']] * (max_length - len(a) - 2)
        for a in collapsed
    ]
)

decode_output = array([
    a +
    [full_dict['<END>']] +
    [full_dict['<PAD>']] * (max_length - len(a) - 1)
    for a in collapsed])

decode_output = reshape(decode_output, (decode_output.shape[0], max_length, 1))

acous_trim = delete(acoustics, exclude_indices[0],0)



# Model Construction
embed_dim = 20
hidden_dim = 1000
adjusted_hd = frames * ceil(1000 / frames)

# Shortened VGG Block
acous_input = Input(shape=(frames, features, 1))
net = Conv2D(64, kernel_size=(3, 3))(acous_input)
net = Conv2D(64, kernel_size=(3, 3))(net)
net = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(net)
net = Conv2D(128, kernel_size=(3, 3))(net)
net = Conv2D(128, kernel_size=(3, 3))(net)
net = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(net)
net = Conv2D(256, kernel_size=(3, 3))(net)
net = Conv2D(256, kernel_size=(3, 3))(net)
net = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(net)
net = Flatten()(net)
net = Dense(adjusted_hd)(net)
encode_input = Reshape((int(adjusted_hd / embed_dim), embed_dim))(net)

# Transformer Block (Output 1D segment prediction tensor)
model = get_model_ne(
    token_num=unique_segments,
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
    embed_trainable=True,
    embed_weights=random((unique_segments, embed_dim))
)

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', 'crossentropy'])

reflatten = [item for sublist in collapsed for item in sublist]
class_weights = hstack((array([1, 1, 1]), class_weight.compute_class_weight('balanced', classes=unique(reflatten), y=reflatten))).tolist()

def weightedloss(weights,depth):

    def loss(y_true,y_pred):
        y_true_hot = tf.one_hot(tf.cast(y_true, 'int32'), int(depth))
        class_weights = tf.constant([weights])
        loss_weights = tf.reduce_sum(class_weights*y_true_hot, axis=1)
        unweighted_losses = tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred)
        weighted_losses = unweighted_losses*loss_weights
        return weighted_losses
        #return tf.reduce_mean(weighted_losses)
    return loss

model.compile(optimizer='adam', loss=weightedloss(class_weights, unique_segments), metrics=['accuracy', 'crossentropy'])




model.fit(
    x=[acous_trim, decode_input],
    y=decode_output,
    epochs=5,
    batch_size=20,
    validation_split=0.1,
)


decoded = decode_ne(
    model,
    acoustics[100:110],
    full_dict['<START>'],
    full_dict['<END>'],
    full_dict['<PAD>'],
)

answer = [[rev_final[a] for a in b] for b in decoded]



