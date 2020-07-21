"""
    TO DO:
"""

#Library
import os, string, scipy.io.wavfile, pickle, csv
from re import sub
from numpy import append, log10, concatenate, vstack
from math import floor, ceil
from python_speech_features import logfbank
from bisect import bisect
from HelperFunctions import empickle, unpickle
import string

#Variables
wlength = HelperFunctions.WLENGTH
wstep = HelperFunctions.WSTEP
filters= HelperFunctions.FILTERS

##Get list of .wav files and associated .phones files in path
def get_files(path, num=None):
    file_list = [f for f in os.listdir(path) if f.endswith(".wav")]
    if num==None:
        num = len(file_list)
    for a in range(num):
        phon_file = file_list[a].replace(".wav", ".phones")
        if os.path.isfile(path+phon_file):
            file_list[a] = (file_list[a], phon_file)
        else:
            print('%s does not exist!' % (path+phon_file))
            file_list[a] = (file_list[a], None)
    ##Returns (wav file, phones file/None)
    return file_list[:num]

##Reads transcription list and returns a dictionary dictionary
def generate_segment_dict(path, phon_file, segdict):
    a =  len(segdict)
    uniques = set(segdict.keys())
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            for b in range(len(transcription)):
                if len(transcription[b]) >2:
                    stripped = sub(r'[^a-z]','', transcription[b][2].lower())
                    if transcription[b][2] not in segdict:
                        if stripped==transcription[b][2] or \
                        transcription[b][2] == "SIL" or \
                        transcription[b][2] == "LAUGH":
                            segdict[transcription[b][2]] = a
                            a+=1
        return segdict
    else:
        return None

##Reads .phones file and returns a transcription list  
def read_in_transcription(path, phon_file, segdict):
    if phon_file != None:
        with open(path+phon_file, "r") as f:
            transcription = [segment.split() for segment in f.readlines()[9:]]
            """for b in range(len(transcription)):
                if len(transcription[b])<3:
                    print transcription[b]"""
        ##Returns (end time, classification [121/2], transcription_index)
        return transcription
    else:
        return None
        
##Reads in wav file, windows, and computes relavant features for audio
def record_features(path,segdict,transcription):
    ##Read in audio file
    print("Reading "+path)
    sampling_rate, wav_array = scipy.io.wavfile.read(path)
    
    ##Find the transcription corresponding to each time step
    seg_times = [float(segment[0]) for segment in transcription if len(segment)>0]
    timepoints = [a*wstep for a in range(0,floor(max(seg_times)/wstep))]    
    slicing = [transcription_list[bisect(seg_times, a)] for a in timepoints]
    segs = [a[2] if len(a)>=3 else "UNKNOWN" for a in slicing]
    
    length = ceil(wlength*sampling_rate)
    feature_list = []
    olist = []
    
    
    ##Calculate filter bank energies for time steps corresponding to existing transcription values
    for a in range(floor(max(seg_times)/wstep)):
        if segs[a] in segdict and sampling_rate*(a*wstep+wlength) < wav_array.shape[0]:
            start = floor(sampling_rate*a*wstep)
            end = floor(sampling_rate*(a*wstep+wlength))
            windowed_sound = wav_array[start:end]
            feature_list.append(logfbank(windowed_sound, sampling_rate, wlength, wstep, filters, length))
            olist.append(segdict[segs[a]])
    
    #Stick input and output values together for output
    f_array = concatenate(feature_list, axis=0)
    return (f_array, olist)

if __name__ == "__main__":
    datapath = "../../Corpora/English, American - Buckeyes Corpus/dataset/"
    segment_dict = "./segment_dict.pkl"
    
    files = get_files(datapath)
    
    ##Dictionary building
    segdict={}
    for file in files:
        segdict = generate_segment_dict(datapath, file[1], segdict)
    print(segdict)
    empickle(segdict, segment_dict)


    ##Input and Output Array Building
    first_time = True
    input = None
    output = []
    for file in files:
        transcription_list= read_in_transcription(datapath, file[1], segdict)
        newin, newout = record_features(datapath+file[0], segdict, transcription_list)
        
        if first_time:
            input = newin
            first_time = False
        else:
            input = vstack((input, newin))
        print(input.shape)
        output.append(newout)
    empickle(input, "./TrainingInput.pickle")
    empickle(output, "./TrainingOutput.pickle")
    

