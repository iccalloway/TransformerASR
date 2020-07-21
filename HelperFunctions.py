from pickle import dump, load
from numpy import array, reshape, pad
from math import ceil

##Variables##

FILTERS = 40
FEATURES = 3*FILTERS
FRAMES = 50
WLENGTH = 0.020
WSTEP = WLENGTH/2
SR = 16000

##Functions##

##Opens pickle file as dictionary
def unpickle(input, default=None):
    try:
        with open(input, "rb") as f:
            item = load(f)
            print("Loaded "+input)
    except IOError:
        if default == None:
            print("No item found.")
            return None
        else:
            print("No item found - creating default...")
            item = default
    return item

##Saves dictionary as pickle file
def empickle(item, output):
    with open(output, "wb") as f:
        dump(item, f)
    print("Saved item to "+output)
    return

##Pad and Reshape to 3d
def prepare_array(arr, frames, final_dim):
    pad_amount = frames*ceil(arr.shape[0]/frames)-arr.shape[0]
    padded = pad(arr, ((0,pad_amount),(0,0)),"constant")
    return padded.reshape(int(padded.shape[0]/frames), frames, final_dim)


def dedupe_adjacent(alist):
     for i in range(len(alist) - 1, 0, -1):
         if alist[i] == alist[i-1]:
             del alist[i]
     return alist