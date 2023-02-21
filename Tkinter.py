#!/usr/bin/env python
# coding: utf-8

# In[730]:


from bertopic import BERTopic
import csv
import gensim
import gensim.downloader
from gensim.models import KeyedVectors
import glob
from io import StringIO
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer 
import numpy as np
import os
from os import path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import pickle
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pydub
from random import sample
import re
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import speech_recognition as sr 
import string
import sys
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import umap
import urllib3
from youtube_dl import YoutubeDL
import youtube_dl


# In[731]:


os.system("sh C:/Users/svcsc/GermanWordEmbeddings-master/GermanWordEmbeddings-master/word2vec_german.sh")


# In[732]:


root = Tk()
root.title("Themensegmentierung")
LARGE_FONT= ("Verdana", 10)
menubar = Menu(root)
root.config(bg='#F6F6F6', menu=menubar)
EXPECTED_SAMPLE_RATE = 16000
audio_downloader = YoutubeDL({"format": "bestaudio"})
pydub.AudioSegment.ffmpeg = "C:/Windows/System32/ffmpeg"
r = sr.Recognizer()
tokenizerReg = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()


# In[733]:


def cleanLemma(pre_data):
    if len(pre_data) == 0:
        predata = "none"
    return pre_data


# In[734]:


def preprocessText(pre_data, newstopwordlist, new_stopwords2):
    pre_data["text_clean"] = pre_data.apply(lambda row: re.sub(r"(\[(.*?)\])", " ", str(row["text"])), axis=1)
    pre_data["text_clean"] = pre_data.apply(lambda row: re.sub(r'[0-9]+', " ", str(row["text_clean"])), axis=1)
    pre_data["tokens"] = pre_data.apply(lambda row: tokenizerReg.tokenize(str(row["text_clean"].lower())), axis=1)
    pre_data["tokens"] = pre_data.apply(lambda row: [element for element in row["tokens"] if element not in newstopwordlist], axis=1)
    pre_data["tokens"] = pre_data.apply(lambda row: [element for element in row["tokens"] if len(element) > 2], axis=1)
    pre_data["token_ner"] = pre_data.apply(lambda row: [element for element in row["tokens"] if element not in new_stopwords2], axis=1)
    pre_data["lemmata"] = pre_data.apply(lambda row: [lemmatizer.lemmatize(word) for word in row["token_ner"]], axis=1)
    return pre_data


# In[735]:


##################################################################
# Speaker Diarization out of Youtube-Video
##################################################################


# In[736]:


def get_check():
    if len(user_input.get())==0:
        messagebox.showwarning('warning', 'You need to file a URL!')
    elif len(selected_language.get()) == 0:
        messagebox.showwarning('warning', 'You need to select a language!')
    else:
        try: 
            output = Label(root, text="download ready", bg='#B2DFEE')
            output.pack()
            get_audio(user_input.get())
        except:
            pass


# In[737]:


def get_audio(idlink):
    try:
        audio_downloader.extract_info(f"{idlink}")
        cut_all_files(path, destination, wav_chunk)
    except youtube_dl.DownloadError as error:
        pass


# In[738]:


def convert_audio_for_model(user_file, output_file='C:/Users/svcsc/Desktop/Code/converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return audio, output_file


# In[739]:


def diarizise(audio_data):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token = "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")
    audio, output = convert_audio_for_model(audio_data)
    diarization = pipeline(output)
    path = "C:/Users/svcsc/Desktop/Code/Speaker/speaker"
    file2write=open(path,'w')
    for turn, _, speaker in diarization.itertracks(yield_label = True):
        splitted_data = f"{turn.start:.0f}, {turn.end:.0f}, {speaker}\n"
        file2write.write(splitted_data)
    file2write.close()
    return audio


# In[740]:


def make_seconds_into_minutes(seconds):
    minutes = 0
    while(seconds > 60):
        seconds = seconds - 60
        minutes += 1
    return minutes, seconds 


# In[741]:


def cut_audio2(data, audio, pointer, startMin, startSek, endMin, endSek, name, wav_chunks):  
    lenght = len(data.index)
    speaker = ""
    chunk = 0
    for line in data.index:
        if(pointer == 0):
            speaker = data.speaker[pointer]
            startMin, startSek = make_seconds_into_minutes(data.start[pointer])
            endMin, endSek = make_seconds_into_minutes(data.stop[pointer])
            pointer += 1
            if(line == lenght-1):
                StartTime = startMin*60*1000+startSek*1000
                EndTime = endMin*60*1000+endSek*1000
                extract = audio[StartTime:EndTime]
                extract.export(f"{wav_chunks}{name}_{str(chunk)}.wav", format="wav")
        else:
            if(speaker != data.speaker[pointer]):
                StartTime = startMin*60*1000+startSek*1000
                EndTime = endMin*60*1000+endSek*1000
                extract = audio[StartTime:EndTime]
                extract.export(f"{wav_chunks}{name}_{str(chunk)}.wav", format="wav")
                chunk += 1
                speaker = data.speaker[pointer]
                startMin, startSek = make_seconds_into_minutes(data.start[pointer])
                endMin, endSek = make_seconds_into_minutes(data.stop[pointer])           
                pointer += 1
                if(line == lenght-1):
                    StartTime = startMin*60*1000+startSek*1000
                    EndTime = endMin*60*1000+endSek*1000
                    extract = audio[StartTime:EndTime]
                    extract.export(f"{wav_chunks}{name}_{str(chunk)}.wav", format="wav")
            else:
                speaker = data.speaker[pointer]
                endMin, endSek = make_seconds_into_minutes(data.stop[pointer])
                pointer += 1
                StartTime = startMin*60*1000+startSek*1000
    EndTime = endMin*60*1000+endSek*1000
    extract = audio[StartTime:EndTime]
    extract.export(f"{wav_chunks}{name}_{str(chunk)}.wav", format="wav")


# In[742]:


def cut_all_files():
    path = "C:/Users/svcsc/"
    destination = "C:/Users/svcsc/Desktop/Code/Speaker/"
    wav_chunks = "C:/Users/svcsc/Desktop/Code/Wave_Chunks/"
    if len(language_cb.get())==0:
        messagebox.showwarning('warning', 'You need to select a language!')
    if language_cb.get() == "en":
        destination = "C:/Users/svcsc/Desktop/Code/Speaker/englisch/"
    else: 
        destination = "C:/Users/svcsc/Desktop/Code/Speaker/deutsch/"
    for filename in glob.glob(os.path.join(path, '*.m4a')):
        try:
            *_, name = filename.split("\\")
            wav_filename = str(f"{destination}{name[0:40]}.wav")
            filename = filename.replace("\\", "/")
            track = AudioSegment.from_file(filename,  format= 'm4a')
            file_handle = track.export(wav_filename, format='wav')
            diarizise(file_handle)
            data = pd.read_csv("C:/Users/svcsc/Desktop/Code/Speaker/speaker", sep=",", header= None)
            data.columns = ["start", "stop", "speaker"]
            track.export(wav_filename, format='mp3')
            cut_audio2(data, track, 0, 0, 0, 0 ,0, name, wav_chunks)
            check_for_transcript()
            output = Label(root, text="speaker diarization ready", bg='#B2DFEE')
            output.pack()
        except BufferError as e:
            print("Error:", str(e))


# In[743]:


##################################################################
# Make transcript
##################################################################


# In[744]:


def cleanRow(pre_data):
    pre_data = pre_data[pre_data.text != ""]
    return pre_data


# In[745]:


def delete_nan(data):
    count_line = 0
    index = []
    for line in data["text"]:
        if (type((line)) == float):
            index.append(count_line)
        count_line += 1
    for indexer in index:
        data = data.drop(index=indexer)
    return data


# In[746]:


def get_short_audio_transcription(path, language):
    with sr.AudioFile(path) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data, language=language)
        except sr.UnknownValueError as e:
            return print("Error:", str(e))
    return text


# In[747]:


def get_large_audio_transcription(path, language):
    sound = AudioSegment.from_file(path)  
    chunks = split_on_silence(sound,
        min_silence_len = 500,
        silence_thresh = sound.dBFS-14,
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened, language=language)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                whole_text += text

    return whole_text


# In[748]:


def get_transcription(path, language):
    sound = AudioSegment.from_file(path)  
    duration = sound.duration_seconds * 1000
    if (duration > 100):
        text = get_large_audio_transcription(path, language)
    else:
        text = get_short_audio_transcription(path, language)
    return text


# In[749]:


def write_wav(file, language, writer):
    text = get_transcription(file, language)
    filepart=open(writer,'a')
    if(type(text) == str and len(text) > 0):
        filepart.write(text)
        filepart.write("\n")
    filepart.close()


# In[750]:


def get_wavs(path, language, writer):
    list_lines = []
    for filename in glob.glob(os.path.join(path, '*.wav')):
        try:
            write_wav(filename, language, writer)
            with open(writer) as f:
                lines = f.readlines()
                list_lines.append(lines)
        except urllib3.exceptions.ProtocolError:
            print(f'Remote disconnected on {filename}')
    return list_lines


# In[751]:


def wav_to_dataframe(path, part_for_dataframe, data_datasatz, language):
    get_wavs(path, language, part_for_dataframe)
    with open(part_for_dataframe) as f:
        lines = f.readlines()
    data = pd.DataFrame(lines, columns=["text"])
    data = cleanRow(data)
    data = delete_nan(data)
    data.to_csv(data_datasatz)


# In[752]:


def check_for_transcript():
    if language_cb.get() != 0:
        language = language_cb.get()
        if language == "de":
            language = "de-DE"
        else:
            language = "en-EN"
    else:
        language = "en-EN" 
    path= "C:/Users/svcsc/Desktop/Code/Wave_Chunks/"
    part_for_dataframe = "C:/Users/svcsc/Desktop/Code/Speaker/transcript_part.txt"
    data_datasatz = "C:/Users/svcsc/Desktop/Code/Speaker/transcript.csv"
    wav_to_dataframe(path, part_for_dataframe, data_datasatz ,language)
    output = Label(root, text="transcript ready", bg='#B2DFEE')
    output.pack()


# In[753]:


##################################################################
# TextTiling
##################################################################


# In[754]:


def check_for_transcript_texttiling():
    if language_cb.get() != 0:
        language = language_cb.get()
        if language == "de":
            language = "de-DE"
            language1 = "german"
            path= "C:/Users/svcsc/Desktop/Code/Speaker/"
            data_datasatz = "C:/Users/svcsc/Desktop/Code/TextTiling/deutsch/transcript.csv"
            part_for_dataframe = "C:/Users/svcsc/Desktop/Code/TextTiling/deutsch/transcript_part.txt"
            name2save = "C:/Users/svcsc/Desktop/Code/TextTiling/deutsch/texttilling.csv"
        else:
            language = "en-EN"
            language1 = "english"
            path= "C:/Users/svcsc/Desktop/Code/Speaker/"
            data_datasatz = "C:/Users/svcsc/Desktop/Code/TextTiling/englisch/transcript.csv"
            part_for_dataframe = "C:/Users/svcsc/Desktop/Code/TextTiling/englisch/transcript_part.txt"
            name2save = "C:/Users/svcsc/Desktop/Code/TextTiling/englisch/texttilling.csv"
    else:
        language = "en-EN" 
    
    wav_to_dataframe(path, part_for_dataframe, data_datasatz ,language)
    output = Label(root, text="transcript ready", bg='#B2DFEE')
    output.pack()
    main(data_datasatz, name2save, language1)
    


# In[755]:


def makeFrameToText(prepath):
    data = pd.read_csv(prepath)
    paragraphs = list(data.text)
    textlist = " ".join(paragraphs)
    return textlist


# In[756]:


def save_txt(text, name):
    txt_filename = f'{str(name)}.txt'
    file = open(txt_filename, w)
    file.write(text)
    file.close()


# In[757]:


def listToString(string):
    str1 = " "
    return (str1.join(string))


# In[758]:


def getSentences(pre_data):
    pre_data["sentences"] = pre_data.parallel_apply(lambda row: sent_tokenize(str(row["text_new"].lower())), axis=1)
    sentences = []
    for parts in pre_data["sentences"]:
        for sentence in parts:
            sentences.append(sentence)
    return sentences


# In[759]:


def vocabulary_introduction(pseudosentences, w):
    newWords1 = set()
    newWords2 = set(pseudosentences[0])
    scores = []
    for token in range(1, len(pseudosentences)-1):
        b1 = set(pseudosentences[token-1]).difference(newWords1)
        b2 = set(pseudosentences[token+1]).difference(newWords2)
        scores.append((len(b1) +len(b2))/(w*2))
        newWords1 = newWords1.union(pseudosentences[token-1])
        newWords2 = newWords2.union(pseudosentences[token+1])

    lastElement = len(set(pseudosentences[len(pseudosentences)-1]).difference(newWords1))
    scores.append(lastElement/(w*2))
    return scores


# In[760]:


def getDepthSideScore(lexScores, currentGap, left):
    depthScore = 0
    i = currentGap
    while lexScores[i] - lexScores[currentGap] >= depthScore:
        depthScore = lexScores[i] - lexScores[currentGap]
        i = i - 1 if left else i + 1
        if (i < 0 and left) or (i == len(lexScores) and not left):
            break
    return depthScore


# In[761]:


def identifyBoundary(lexScores):
    boundaries = []
    depthCutOff = abs(np.mean(lexScores) - np.std(lexScores))
    for currentGap, score in enumerate(lexScores):
        depthLeftScore= getDepthSideScore(lexScores, currentGap, True)
        depthRightScore= getDepthSideScore(lexScores, currentGap, False)
        depthScore = depthLeftScore + depthRightScore
        if depthScore >= depthCutOff:
            boundaries.append(currentGap)
    return boundaries


# In[762]:


def getBoundary(boundary, breaks, w):
    tokenIndices = [w * (gap+1) for gap in boundary]
    sentenceBoundary = set()
    for index in range(len(tokenIndices)):
        sentenceBoundary.add(min(breaks, key = lambda b: abs(b- tokenIndices[index])))
    return sorted(list(sentenceBoundary))


# In[763]:


def getTextParts(sentencesList, boundary):
    cuts = []
    for number in boundary:
        cuts.append(sentencesList[number])
    return cuts


# In[764]:


def cutText(cuts, text):
    substring = []
    newText = []
    for sentence in cuts:
        substring = text.partition(str(sentence))
        print(substring)
        sub = str(substring[0] + substring[1])
        print(sub)
        sublist = []
        sublist.append(sub)
        newText.append(sublist[0])
        text = text.replace(sublist[0], "")
    print(substring[2])
    newText.append(substring[2])
    return newText


# In[765]:


def texttill_this(list_part, language, stopwordlist):  
    w = 20
    sentences = sent_tokenize(list_part) #text in Sätze
    dataSentences = pd.DataFrame(sentences, columns=["text"]) #DataFrame draus machen
    newDataSentences = preprocessText(dataSentences, language, stopwordlist) #Lemmata aus DateFrame draus machen
    sentenceTokens = newDataSentences["lemmata"] #nur Lemmata raussuchen
    lemmata = sentenceTokens.values.tolist() #Lemmata-Liste
    vocabularyIntroduction = vocabulary_introduction(lemmata, w)
    boundary = identifyBoundary(vocabularyIntroduction)
    textParts = getTextParts(sentences, boundary) #Sätze herausfinden, bei denen die thematische Grenze liegt
    newText = cutText(textParts, list_part) #den Text nach diesen Sätzen aufteilen
    return newText


# In[766]:


def makeFrameFromText(predata):
    with open(predata) as file:
        line_list =  (file.read())
    return line_list


# In[767]:


def splitTextinBytes(list2split, language, stopwordlist):
    newString = ""
    size = 0
    newData = pd.DataFrame()
    for elements in list2split:
        setsize = sys.getsizeof(elements)
        newString += elements
        size += setsize
        if((size > 170000 and elements == ".") or (size > 170000 and elements == "?")):
            newText = texttill_this(newString, language, stopwordlist)
            data = pd.DataFrame(newText)
            newData = pd.concat([newData, data], axis=0)
            newString = ""
            size = 0
    newText = texttill_this(newString, language, stopwordlist)
    data = pd.DataFrame(newText)
    newData = pd.concat([newData, data], axis=0)
    newData.rename( columns={0 :'text'}, inplace=True )
    index_range = [*range(0, len(newData), 1)]
    newData.index = index_range
    return newData


# In[768]:


def workFrame(frame, language, stopwordlist):
    newData = cleanRow(frame)
    newData = delete_nan(newData)
    newData=preprocessText(newData, language, stopwordlist)
    newData["forvector"] = newData.apply(lambda row: str(row["lemmata"]), axis=1)
    return newData


# In[769]:


def saveFiles(name, file):
    try:
        csv_filename = f'{str(name)}.csv'
        file.to_csv(csv_filename)
    except KeyError:
        print("Saving Data failed.")


# In[770]:


def getMoreStopWords(data2use):
    moreStopWords = pd.read_csv(data2use, sep=" ")
    moreStopWords = list(moreStopWords.reuters)
    return moreStopWords


# In[771]:


def getMoreStopWordsGerman(data2use):
    moreStopWords_german = makeFrameFromText(data2use)
    moreStopWords_german = moreStopWords_german.split("\n")
    return moreStopWords_german


# In[772]:


def makeStopwords_eng(extrawords):
    stop_words = (stopwords.words("english"))
    final_words = stop_words + extrawords
    return final_words


# In[773]:


def makeStopwords_ger(extrawords):
    stop_words = (stopwords.words("german"))
    final_words = stop_words + extrawords
    return final_words


# In[774]:


def main(data2use, name2save, language):
    english_stopwords = "C:/Users/svcsc/Desktop/Code/Stopwörter/pyrouge.txt"
    german_stopword = "C:/Users/svcsc/Desktop/Code/Stopwörter/deutsche_stopwoerter.txt"
    if(language == "english"):
        moreStopWords = getMoreStopWords(english_stopwords)
        stopwordlist = makeStopwords_eng(moreStopWords)
    if(language == "german"):
        moreStopWords = getMoreStopWordsGerman(german_stopword)
        stopwordlist = makeStopwords_ger(moreStopWords)
    line_list = makeFrameFromText(data2use)
    nextTry = splitTextinBytes(line_list, language, stopwordlist)
    newData = workFrame(nextTry, language, stopwordlist)
    saveFiles(name2save, newData)
    return newData


# In[775]:


def check_tt():
    if language_cb.get() != 0:
        language = language_cb.get()
        if language == "de":
            language = "german"
            path= "C:/Users/svcsc/Desktop/Code/Speaker/deutsch/transcript.csv"
        else:
            language = "english"
            path= "C:/Users/svcsc/Desktop/Code/Speaker/englisch/transcript.csv"
    part_for_dataframe = "C:/Users/svcsc/Desktop/Code/TextTiling/texttiling"
    main(path, part_for_dataframe, language)
    output = Label(root, text="testtiling ready", bg='#B2DFEE')
    output.pack()
    


# In[776]:


##################################################################
# markov english
##################################################################


# In[777]:


def get_classes(data):
    entertainment = data[data["label"] == "entertainment"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_entertainment.txt", header=False, index=False)
    sport = data[data["label"] == "sport"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_sport.txt", header=False, index=False)
    tech = data[data["label"] == "tech"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_tech.txt", header=False, index=False)
    business = data[data["label"] == "business"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_business.txt", header=False, index=False)
    politics = data[data["label"] == "politics"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_politics.txt", header=False, index=False)
    classes_list = ["entertainment", "sport", "tech", "business", "politics"]

    input_files = [
        "C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_entertainment.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_sport.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_tech.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_business.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Englisch/train_politics.txt",
    ]
    return input_files, classes_list


# In[778]:


def get_input_and_labels(input_files):
    input_text = []
    labels = []
    for label, f in enumerate(input_files):
        for line in open(f, encoding='utf-8'):
            line = line.rstrip().lower()
            if line:
                line = line.translate(str.maketrans("", "", string.punctuation))
                input_text.append(line)
                labels.append(label)
    return input_text, labels


# In[779]:


def convert_data_to_integer(train_text, test_text):
    idx = 1
    word2idx = {"<unk>":0}

    for text in train_text:
        tokens = text.split()
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

    train_text_int = []
    test_text_int = []

    for text in train_text:
        tokens = text.split()
        line_as_int = [word2idx[token] for token in tokens]
        train_text_int.append(line_as_int)
        
    for text in test_text:
        tokens = text.split()
        line_as_int = [word2idx.get(token,0) for token in tokens]
        test_text_int.append(line_as_int)
    return word2idx, train_text_int, test_text_int


# In[780]:


def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None: #beginning
                pi[idx] += 1
            else: #transition
                A[last_idx, idx] += 1
            last_idx = idx


# In[781]:


def get_class_information(text_int, Yvalue, word2id, n):
    N = len(word2id)
    A = np.ones((N,N)) #add one smoothing
    pi = np.ones(N)
    compute_counts([t for t, y in zip(text_int, Yvalue) if y==n], A, pi) 
    
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()
    logA = np.log(A)
    logpi = np.log(pi)
    
    count = sum(y == n for y in Yvalue)
    total = len(Yvalue)
    p = count / total   #compute the prior probabilities
    logp = np.log(p)
    return A, pi, logA, logpi, p, logp


# In[782]:


def get_labels_to_classes(input_text, Ytrain, classes_list, predicted, save):
    my_dict = {}
    for label2, f in enumerate(classes_list):
        my_dict[str(label2)] = (str(classes_list[label2]))
    predict = predicted.astype(int)
    data = pd.DataFrame()
    data["text"] = input_text
    data["actual_labels"] = Ytrain
    data["predicted_labels"] = (predict)
    data["old_label"] = data.apply(lambda row: my_dict.get(str(row["actual_labels"]), my_dict), axis=1)
    data["new_label"] = data.apply(lambda row: my_dict.get(str(row["predicted_labels"]), my_dict), axis=1)
    data.to_csv(save)
    return data, my_dict


# In[783]:


def get_confusion_matrix_plot(conf, save):
    plt.figure(figsize = (10,8))
    heatmap = sns.heatmap(conf/np.sum(conf), annot=True, fmt='.1%', cmap='YlGnBu')
    heatmap.set(title='Confusion Matrix')
    heatmap.set(xlabel='Predicted', ylabel='Actual')
    plt.savefig(save)


# In[784]:


class CLassifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K = len(logpriors) # number of classes

    def _compute_log_likelihood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]
        
        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]
            last_idx = idx
        return logprob
    
    def predict(self, inputs):
        prediction = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            posteriors = [self._compute_log_likelihood(input_,c)+ self.logpriors[c] \
                          for c in range (self.K)]
            pred = np.argmax(posteriors)
            prediction[i] = pred
        return prediction


# In[785]:


def convert_new_data_to_integer(new_text):
    idx = 1
    word2idx = {"<unk>":0}

    for text in new_text:
        tokens = text.split()
        for token in tokens:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

    train_text_int = []

    for text in new_text:
        tokens = text.split()
        line_as_int = [word2idx[token] for token in tokens]
        train_text_int.append(line_as_int)
    return train_text_int


# In[786]:


def new_labels_to_classes(input_text, classes_list, predicted, save):
    my_dict = {}
    for label2, f in enumerate(classes_list):
        my_dict[str(label2)] = (str(classes_list[label2]))
    predict = predicted.astype(int)
    data = pd.DataFrame()
    data["text"] = input_text
    data["predicted_labels"] = (predict)
    data["labels"] = data.apply(lambda row: my_dict.get(str(row["predicted_labels"]), my_dict), axis=1)
    data.drop('predicted_labels', axis=1, inplace=True)
    data = cleanRow(data)
    data = delete_nan(data)
    data.to_csv(save)
    return data, my_dict


# In[787]:


def check_markov(data):
    bbc = "C:/Users/svcsc/Desktop/Code/bbc_articles.csv"
    new_save = "C:/Users/svcsc/Desktop/Code/Markov/Englisch/Ergebnisse/topics.csv"
    new_save_train = "C:/Users/svcsc/Desktop/Code/Markov/Englisch/articles_train.csv"
    new_save_test = "C:/Users/svcsc/Desktop/Code/Markov/Englisch/articles_test.csv"
    
    bbc_articles = pd.read_csv(bbc, index_col=0)
    input_files, classes_list = get_classes(bbc_articles)
    input_text, labels = get_input_and_labels(input_files)
    train_text, test_text, Ytrain, Ytest = train_test_split(input_text, labels)
    word2id, train_text_int, test_text_int = convert_data_to_integer(train_text, test_text)

    A0, pi0, logA0, logpi0, p0, logp0 = get_class_information(train_text_int, Ytrain, word2id, n=0)
    A1, pi1, logA1, logpi1, p1, logp1 = get_class_information(train_text_int, Ytrain, word2id, n=1)
    A1, pi1, logA1, logpi1, p1, logp1 = get_class_information(train_text_int, Ytrain, word2id, n=1)
    A2, pi2, logA2, logpi2, p2, logp2 = get_class_information(train_text_int, Ytrain, word2id, n=2)
    A3, pi3, logA3, logpi3, p3, logp3 = get_class_information(train_text_int, Ytrain, word2id, n=3)
    A4, pi4, logA4, logpi4, p4, logp4 = get_class_information(train_text_int, Ytrain, word2id, n=4)

    clf = CLassifier([logA0, logA1, logA2, logA3, logA4], [logpi0, logpi1, logpi2, logpi3, logpi4], [logp0, logp1, logp2, logp3, logp4])
    Ptrain = clf.predict(train_text_int)
    Ptest = clf.predict(test_text_int)

    data_train, my_dict = get_labels_to_classes(train_text, Ytrain, classes_list, Ptrain, new_save_train)
    data_test, my_dict = get_labels_to_classes(test_text, Ytest, classes_list, Ptest, new_save_test)

    data_without_topics = pd.read_csv(data, index_col=0)
    punctlist = list(data_without_topics['text']) 
    puncttext = ' '.join(str(element) for element in punctlist)
    sentence_text = sent_tokenize(puncttext)
    new_text_int = convert_new_data_to_integer(punctlist)

    Pnew = clf.predict(new_text_int)
    data_new, my_dict_new = new_labels_to_classes(punctlist, classes_list, Pnew, new_save)
    return data_new


# In[788]:


##################################################################
# Markov German
##################################################################


# In[789]:


def get_classes_german(data):
    Etat = data[data["label"] == "Etat"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Etat.txt", header=False, index=False)
    Inland = data[data["label"] == "Inland"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Inland.txt", header=False, index=False)
    International = data[data["label"] == "International"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/International.txt", header=False, index=False)
    Kultur = data[data["label"] == "Kultur"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Kultur.txt", header=False, index=False)
    Panorama = data[data["label"] == "Panorama"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Panorama.txt", header=False, index=False)
    Sport = data[data["label"] == "Sport"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Sport.txt", header=False, index=False)
    Web = data[data["label"] == "Web"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Web.txt", header=False, index=False)
    Wirtschaft = data[data["label"] == "Wirtschaft"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Wirtschaft.txt", header=False, index=False)
    Wissenschaft = data[data["label"] == "Wissenschaft"].drop(columns=["label"]).to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Wissenschaft.txt", header=False, index=False)
    
    classes_list = ['Etat', 'Inland', 'International', 'Kultur', 'Panorama', 'Sport', 'Web', 'Wirtschaft', 'Wissenschaft']

    input_files = [
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Etat.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Inland.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/International.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Kultur.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Panorama.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Sport.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Web.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Wirtschaft.txt",
        "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Wissenschaft.txt",
    ]
    return input_files, classes_list


# In[790]:


def check_markov_german(data):
    new_save = "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Ergebnisse/topics.csv"
    articles = "C:/Users/svcsc/Desktop/Code/Ten_Thousand_german_articles.csv"
    new_save_train = "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/articles_train.csv"
    new_save_test = "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/articles_test.csv"
    german_articles = pd.read_csv(articles, index_col=0)
    german_frac = german_articles.sample(frac = 0.02)

    input_files, classes_list = get_classes_german(german_frac)
    input_text, labels = get_input_and_labels(input_files)
    train_text, test_text, Ytrain, Ytest = train_test_split(input_text, labels)
    word2id, train_text_int, test_text_int = convert_data_to_integer(train_text, test_text)
    
    A0, pi0, logA0, logpi0, p0, logp0 = get_class_information(train_text_int, Ytrain, word2id, n=0)
    A1, pi1, logA1, logpi1, p1, logp1 = get_class_information(train_text_int, Ytrain, word2id, n=1)
    A2, pi2, logA2, logpi2, p2, logp2 = get_class_information(train_text_int, Ytrain, word2id, n=2)
    A3, pi3, logA3, logpi3, p3, logp3 = get_class_information(train_text_int, Ytrain, word2id, n=3)
    A4, pi4, logA4, logpi4, p4, logp4 = get_class_information(train_text_int, Ytrain, word2id, n=4)
    A5, pi5, logA5, logpi5, p5, logp5 = get_class_information(train_text_int, Ytrain, word2id, n=5)
    A6, pi6, logA6, logpi6, p6, logp6 = get_class_information(train_text_int, Ytrain, word2id, n=6)
    A7, pi7, logA7, logpi7, p7, logp7 = get_class_information(train_text_int, Ytrain, word2id, n=7)
    A8, pi8, logA8, logpi8, p8, logp8 = get_class_information(train_text_int, Ytrain, word2id, n=8)             
             
    clf = CLassifier([logA0, logA1, logA2, logA3, logA4, logA5, logA6, logA7, logA8], [logpi0, logpi1, logpi2, logpi3, logpi4, logpi5, logpi6, logpi7, logpi8], [logp0, logp1, logp2, logp3, logp4, logp5, logp6, logp7, logp8])
    Ptrain = clf.predict(train_text_int)
    Ptest = clf.predict(test_text_int)
    data_train, my_dict = get_labels_to_classes(train_text, Ytrain, classes_list, Ptrain, new_save_train)
    data_test, my_dict = get_labels_to_classes(test_text, Ytest, classes_list, Ptest, new_save_test)
    
    data_without_topics = pd.read_csv(data, index_col=0)
    
    data_without_topics = delete_nan(data_without_topics)
    new_text_list = data_without_topics.text.to_list()
    new_text_int = convert_new_data_to_integer(new_text_list)
    Pnew = clf.predict(new_text_int)
    data_new, my_dict_new = new_labels_to_classes(new_text_list, classes_list, Pnew, new_save)
    return data_new


# In[791]:


##################################################################
# Snorkel
##################################################################


# In[792]:


def lower(pre_data):
    pre_data["text"] = pre_data.apply(lambda row: (str(row["text"].lower())), axis=1)
    return pre_data


# In[793]:


def get_category(new_label):
    switch={
        1: "entertainment",
        2: "sport",
        3: "tech",
        4: "business",
        5: "politics",
        #6: "climate",
        #7: "medicine"
    }
    return switch.get(new_label, "none")


# In[794]:


def get_category_german(new_label):
    switch={
        1: "Etat",
        2: "Inland",
        3: "International",
        4: "Kultur",
        5: "Panorama",
        6: "Sport",
        7: "Web",
        8: "Wirtschaft",
        9: "Wissenschaft"
    }
    return switch.get(new_label, "Unknown")


# In[795]:


def make_text_again(data, text):
    f = open(text, "w")
    f.write(data)
    f.close()


# In[796]:


def get_snorkel(transcript):
    ENTERTAINMENT = 1
    SPORT = 2
    TECH = 3
    BUSINESS = 4
    POLITICS = 5
    CLIMATE = 6
    MEDICINE = 7
    ABTAIN = -1

    @labeling_function()
    def lf_regex_contains_movie(x):
        return ENTERTAINMENT if(re.search(r"film|sendung|tv|preis|drama|comedy|entertain|kino|schauspieler|schauspielerin|skript|regisseur", (str(x.text)))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_celebrity(x):
        return ENTERTAINMENT if(re.search(r"star|sternchen|show|promi|autor|theater|oscar|nominiert", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_music(x):
        return ENTERTAINMENT if(re.search(r"music|musical|promi|rock|pop|jazz|band|singer|song|single|sänger|artist|sängerin|charts|show", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_movies(x):
        return ENTERTAINMENT if(re.search(r"buffy|wonder woman|hobbit|hollywood|kassenschlager", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_regie(x):
        return ENTERTAINMENT if(re.search(r"Peter Jackson|wonder woman|hobbit|lod of the rings", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_sports(x):
        return SPORT if(re.search(r"sport|gewinnen|finale|spiel|wettkampf|wettstreit|sportler|sportlerin|coach|MBA|fußball|athlet", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_cups(x):
        return SPORT if(re.search(r"worlds cup|bundesliga|wm|em|titelverteidiger|olympia|kampf|medallie|gold|silver|bronze|platz|sieg|rekord", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_sportacts(x):
        return SPORT if(re.search(r"spiel|gegeneinander|laufen|schwimmen|halbfinale|springer|fußballer|tennis|handball|hürdenlauf|formel1|rennfahrer|faustball|volleyball|tanzen|springen|biathlon", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_tech(x):
        return TECH if(re.search(r"technologie|technik|online|mobile|digital", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_web(x):
        return TECH if(re.search(r"web|html|code|net|spam|spyware", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_games(x):
        return TECH if(re.search(r"game|orpg|xbox|gaming|console", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_devices(x):
        return TECH if(re.search(r"gameboy|smartphone|notebook|playstation|broadbank|e-mail", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_it(x):
        return TECH if(re.search(r"it|computer|virus|programmieren|blogger|power|investition|intel|microsoft|fiat|telecom|ericsson|apple", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_stock(x):
        return BUSINESS if(re.search(r"dollar|euro|stock|steig|fall|rate|DAX|defizit|deal|gewinn|verlust|konsumen", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_company(x):
        return BUSINESS if(re.search(r"besitzer|firma|unternehmen|CEO|boss|chef|profit|innovation|sale|verkauf|anteil|versicherung", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_business(x):
        return BUSINESS if(re.search(r"unternehmen|asset|bank|payout|marketing|job|business|manager|office|suppliers|supplier", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_economy(x):
        return BUSINESS if(re.search(r"wirtschaft|preis|markt|geld|erhöhen|export|inport", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_politic(x):
        return POLITICS if(re.search(r"konservativ|kandidat|demokrat|partei|power|politik|liberal|regierung|linke|rechte|spd|cdu|grüne|fdp|afd|terror", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_election(x):
        return POLITICS if(re.search(r"kommision|bundeskanzler|wahlkampf|minister|präsident|politiker|krise|wahl|wählen|kampagne", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_president(x):
        return POLITICS if(re.search(r"präsidentschaft|vorsitzender|gericht|fall|protest|demonstration|krieg|partei|wählen|skandal|regel|product|industrie|hartz|mindestlohn|pauschale|steuern|agenda|gesetz", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_climate(x):
        return CLIMATE if(re.search(r"klima|erwärmung|global warming|klimawandel|wetter|athmosphäre|gesetz|fridays for future|klimademo|letzte generation", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_nature(x):
        return CLIMATE if(re.search(r"tsunami|huricane|plastik|ocean|wasserspiegel|sturm|windhose|wirbelsturm|vulkan|erdbeben", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_environment(x):
        return CLIMATE if(re.search(r"temperatur|terrain|biosphäre|greenhouse|environment|umwelt|eiszeit|waldbrand", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_covid(x):
        return MEDICINE if(re.search(r"impfung|covid|impfen|impfstoff|nebenwirkungen|corona|krank|triage", str(x.text))) else ABTAIN

    @labeling_function()
    def lf_regex_contains_medizin(x):
        return MEDICINE if(re.search(r"grippe|fieber|virus|hospital|krankenhaus|krankenschwester|pflege|krankenstation|drogen|doctor|arzt|pflegen|gesundheit|health care|schmerzmittel", str(x.text))) else ABTAIN

    lfs = [lf_regex_contains_movie, lf_regex_contains_celebrity,lf_regex_contains_music, lf_regex_contains_movies, lf_regex_contains_regie, lf_regex_contains_sports, lf_regex_contains_cups, lf_regex_contains_sportacts, lf_regex_contains_tech, lf_regex_contains_web, lf_regex_contains_games, lf_regex_contains_devices, lf_regex_contains_it, lf_regex_contains_stock, lf_regex_contains_company, lf_regex_contains_business, lf_regex_contains_economy, lf_regex_contains_politic, lf_regex_contains_election, lf_regex_contains_president, lf_regex_contains_climate, lf_regex_contains_nature, lf_regex_contains_environment,lf_regex_contains_covid, lf_regex_contains_medizin]
    applier = PandasLFApplier(lfs=lfs)
        
    bbc = ("C:/Users/svcsc/Desktop/Code/bbc_articles.csv")
    english_articles = pd.read_csv(bbc, index_col=0)
    df = cleanRow(english_articles)
    df["text"] = df.apply(lambda row: str(row['text']).lower(), axis=1)

    train_text, test_text = train_test_split(df, train_size=0.9)
    Ltrain = applier.apply(df=train_text)
    label_model = LabelModel(cardinality=10, verbose=True)
    label_model.fit(L_train=Ltrain, n_epochs=500, log_freq=100,seed=123)
    
    label_proba_train = label_model.predict_proba(Ltrain)
    most_likely_label_train = np.argmax(label_proba_train, axis = 1)
    most_likely_label_train = most_likely_label_train.astype(int)
    train_text["new_labels"] = most_likely_label_train
    train_text["labels"] = train_text.apply(lambda row: get_category(row["new_labels"]), axis=1)
    train_text = train_text.drop(columns=["new_labels"])
    train_text.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Englisch/articles_train.csv")
    
    Ltest = applier.apply(df=test_text)
    label_proba_test = label_model.predict_proba(Ltest)
    most_likely_label_test = np.argmax(label_proba_test, axis = 1)
    most_likely_label_test = most_likely_label_test.astype(int)
    test_text["new_labels"] = most_likely_label_test
    test_text["labels"] = test_text.apply(lambda row: get_category(row["new_labels"]), axis=1)
    test_text = test_text.drop(columns=["new_labels"])
    test_text.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Englisch/articles_test.csv")

    test = pd.read_csv(transcript, index_col=0)

    Ltages = applier.apply(df=test)
    label_proba = label_model.predict_proba(Ltages)
    most_likely_label = np.argmax(label_proba, axis = 1)
    most_likely_label = most_likely_label.astype(int)
    test["new_labels"] = most_likely_label
    test["labels"] = test.apply(lambda row: get_category(row["new_labels"]), axis=1)
    test = test.drop(columns=["new_labels"])
    return  test    


# In[797]:


def get_snorkel_german(transcript):
    ETAT = 1
    INLAND = 2
    INTERNATIONAL = 3
    KULTUR = 4
    PANORAMA = 5
    SPORT = 6
    WEB = 7
    WIRTSCHAFT = 8
    WISSENSCHAFT = 9

    @labeling_function()
    def lf_regex_contains_wissenschaft(x):
        return WISSENSCHAFT if(re.search(r"wissenschaft|science|forschung|universität|hochschule|forscher|experiement|studie|applied|test|wissenschaften|dekan", (str(x.text)))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_fachschaften(x):
        return WISSENSCHAFT if(re.search(r"chemie|physik|mathematik|reagenzglas|genetik|virologie|blindstudie|doppelblind", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_studien(x):
        return WISSENSCHAFT if(re.search(r"mittelwert|mean|varianz|teilnehmer|artikel|doktor|professor|stipendium|naturwissenschaften|literaturwissenschaft|sozialwissenschaft|sprachwissenschaft|Overfitting", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_regie(x):
        return KULTUR if(re.search(r"Peter Jackson|wonder woman|hobbit|lod of the rings", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_movie(x):
        return KULTUR if(re.search(r"film|sendung|tv|preis|drama|comedy|entertain|kino|schauspieler|schauspielerin|skript|regisseur",
    (str(x.text)))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_celebrity(x):
        return KULTUR if(re.search(r"star|sternchen|show|promi|autor|theater|oscar|nominiert",
    str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_music(x):
        return KULTUR if(re.search(r"music|musical|promi|rock|pop|jazz|band|singer|song|single|sänger|artist|sängerin|charts|show",
    str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_movies(x):
        return KULTUR if(re.search(r"buffy|wonder woman|hobbit|hollywood|kassenschlager", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_sports(x):
        return SPORT if(re.search(r"sport|win|final|match|challenge|sportler|MBE|coach|competition|opener|athlet|screen", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_cups(x):
        return SPORT if(re.search(r"worlds cup|cup|title|olymp|battle|medal|gold|silver|victory|record", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_sportacts(x):
        return SPORT if(re.search(r"play|compete|run|swim|final|hurdler|soccer|dop|marathon|football|champion|tennis|rugby|dance|curling|biathlon", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_tech(x):
        return WEB if(re.search(r"technology|tech|online|mobile|digital", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_web(x):
        return WEB if(re.search(r"web|html|code|net|spam|spyware", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_games(x):
        return WEB if(re.search(r"game|orpg|xbox|gaming|console", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_devices(x):
        return WEB if(re.search(r"gameboy|smartphone|notebook|playstation|broadbank|e-mail", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_it(x):
        return WEB if(re.search(r"IT|computer|virus|programming|blogger|power|investment|intel|microsoft|fiat|telecom|ericsson|apple", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_it2(x):
        return WEB if(re.search(r"web|bausatz|smartphone|open-source|chip|modem|internetzugriff|raspberry pi|vernetzung|google|3d-drucker|iphone|apple|surface|update|software|elektroauto|autopilot|tesla|elektronik|hd|akku|it|hp|generation|chrome|os celeron|windows|webseite|clients|tablet|notebook|geräte|datenbank|ios|stromverbrauch|prozessor|prozessorhersteller|qualcomm|performance|roboter|prototyp|programmierer|programmiersprache|python|java|kotlin|minirechner", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_stock(x):
        return WIRTSCHAFT if(re.search(r"dollar|euro|stock|gain|drop|rate|DAX|deficit|deal|growth|consumer", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_company(x):
        return WIRTSCHAFT if(re.search(r"owner|firm|company|CEOchef|profit|innovation|sale|share|insurance", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_business(x):
        return WIRTSCHAFT if(re.search(r"compan|shoppers|asset|bank|payout|marketing|job|business|manager|office|suppliers|supplier", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_economy(x):
        return WIRTSCHAFT if(re.search(r"economy|price|market|money|rise|export|inport", str(x.text))) else PANORAMA


    @labeling_function()
    def lf_regex_contains_inland(x):
        return INLAND if(re.search(r"rot|schwarz|volk|sozialdemokrat|wahl|bundespräsident|umfragen|repubik|österreich|österreichisch|fpö|parlament|christen partei|wahlkampf|wiederwahl|stichwahl|niederlage|korruptionsstaatsanwaltschaft|innenministerium|villach|verwaltungsleiter|landespolizei|rechtsextrem|steiermark|vorarlberg|finanzministerium|regierungsprogramm|övp|burgenland|eisenstadt|regierungsamtlich|wien", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_bund(x):
        return INLAND if(re.search(r"bundeskanzler|bundestag|bundesland|ministerpräsident|merkel|scholz|schröder|wahl|deutsche bahn|wahlkampf", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_parteien(x):
        return INLAND if(re.search(r"spd|cdu|union|övp|kanzler|altkanzler|fdp|linke|rechte|npd|basis|die grüne|landtag|bundesrat|bundesebene|landesebene|hartz|gremium|berlin", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_international(x):
        return INTERNATIONAL if(re.search(r"usa|uk|europa|asien|nato|russland|gaza|südamerika", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_ausland(x):
        return INTERNATIONAL if(re.search(r"deutschland|schleswig-holstein|hamburg|bremen|niedersachsen|sachsen|sachsen-anhalt|mecklemburg vorpommern|hessen|baden würtemberg|bayern|nordrhein", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_international2(x):
        return INTERNATIONAL if(re.search(r"arabisch|is|golfkooperationsrat|golfstaaten|damaskus|gefechte|terrormiliz|grenze|türkei|kämpfe|grenzprovinz|grenzregion|rebellen|syrien|washington|usa|merkel|deutschland|auswärtiges|US|generalsekräter|bürgerkrieg|vereinte nationen|new york|madrid|berlin|paris|frankreich|griechenland", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_interpol(x):
        return INTERNATIONAL if(re.search(r"republikaner|liberale|international|uno|welt|nordkoreo|konflikt|bürgerkrieg", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_laender(x):
        return INTERNATIONAL if(re.search(r"japan|china|nahost|iran|irak|griechenland|flüchtlingswelle", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_etat(x):
        return ETAT if(re.search(r"agentur|ard|zdf|wdr|öffentlich rechtliches|radio|mdr|nachrichten|zeitungen", str(x.text))) else PANORAMA

    @labeling_function()
    def lf_regex_contains_etat2(x):
        return ETAT if(re.search(r"quotenregelung|gleichstellung|ard|quote|nutzung|einschaltquoten|betriebsrat|gehalt|gehaltsanpassung|personal|personalabbau|kündigung|personalerkrankung|redaktion|jobwechsel|presse|kommuniktionsbranche|medien|medienwelt|etat|plattform|leser|herausgeber|nachrichten|autorenkollektiv|reporter|betrieb|investment|journalist|news|app|plattform|kommentar|moderator|shitstorm|facebook|werbung|user|korrespondent|community|kurznachricht|programm|fernsehen|radio|kommunikation|programmrichtlinien|zentralbetriebsrat|zukunft|zukunftskurs|mediensprecher|funkhaus|sendung|forum|reportage|publikum", str(x.text))) else PANORAMA

    lfs = [lf_regex_contains_wissenschaft, lf_regex_contains_fachschaften,lf_regex_contains_studien,lf_regex_contains_regie, lf_regex_contains_movie, lf_regex_contains_celebrity, lf_regex_contains_music, lf_regex_contains_movies, lf_regex_contains_sports, lf_regex_contains_cups, lf_regex_contains_sportacts, lf_regex_contains_tech, lf_regex_contains_web, lf_regex_contains_games, lf_regex_contains_devices, lf_regex_contains_it, lf_regex_contains_it2, lf_regex_contains_stock, lf_regex_contains_company, lf_regex_contains_business, lf_regex_contains_economy, lf_regex_contains_inland, lf_regex_contains_bund, lf_regex_contains_parteien, lf_regex_contains_international, lf_regex_contains_ausland, lf_regex_contains_international2, lf_regex_contains_interpol, lf_regex_contains_laender, lf_regex_contains_etat, lf_regex_contains_etat2]
    applier = PandasLFApplier(lfs=lfs)

    articles = ("C:/Users/svcsc/Desktop/Code/Ten_Thousand_german_articles.csv")
    german_articles = pd.read_csv(articles, index_col=0)
    df = cleanRow(german_articles)
    df["text"] = df.apply(lambda row: str(row['text']).lower(), axis=1)

    train_text, test_text = train_test_split(df, train_size=0.9)
    Ltrain = applier.apply(df=train_text)
    label_model = LabelModel(cardinality=10, verbose=True)
    label_model.fit(L_train=Ltrain, n_epochs=500, log_freq=100,seed=123)
    
    label_proba_train = label_model.predict_proba(Ltrain)
    most_likely_label_train = np.argmax(label_proba_train, axis = 1)
    most_likely_label_train = most_likely_label_train.astype(int)
    train_text["new_labels"] = most_likely_label_train
    train_text["labels"] = train_text.apply(lambda row: get_category(row["new_labels"]), axis=1)
    train_text = train_text.drop(columns=["new_labels"])

    train_text["new_labels"] = label_model.predict(L=Ltrain, tie_break_policy="panorama")
    train_text["category"] = train_text.apply(lambda row: get_category_german(row["new_labels"]), axis=1)
    train_text.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Deutsch/articles_train.csv")
    
    Ltest = applier.apply(df=test_text)
    label_proba_test = label_model.predict_proba(Ltest)
    most_likely_label_test = np.argmax(label_proba_test, axis = 1)
    most_likely_label_test = most_likely_label_test.astype(int)
    test_text["new_labels"] = most_likely_label_test
    test_text["labels"] = test_text.apply(lambda row: get_category_german(row["new_labels"]), axis=1)
    test_text = test_text.drop(columns=["new_labels"])
    test_text.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Deutsch/articles_test.csv")

    tagesschau = pd.read_csv(transcript, index_col=0)

    Ltages = applier.apply(df=tagesschau)
    label_proba = label_model.predict_proba(Ltages)
    most_likely_label = np.argmax(label_proba, axis = 1)
    most_likely_label = most_likely_label.astype(int)
    tagesschau["new_labels"] = most_likely_label
    tagesschau["labels"] = tagesschau.apply(lambda row: get_category_german(row["new_labels"]), axis=1)
    tagesschau = tagesschau.drop(columns=["new_labels"])
    return  tagesschau


# In[798]:


##################################################################
# BERTopic
##################################################################


# In[799]:


def make_model(language):
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=100)
    topic_model = BERTopic(umap_model=umap_model, language = language, calculate_probabilities=True)
    return topic_model


# In[800]:


def get_topics_and_probs(topic_model, data):
    topics, probabilities = topic_model.fit_transform(data)
    return topics, probabilities


# In[801]:


def get_topic_prediction(topic_model, data):
    topic_prediction = topic_model.topics_[:]# Save the predictions in the dataframe
    data['topic_prediction'] = topic_prediction
    data = data.drop(columns= ["text_clean", "tokens", "token_ner", "lemmata"])
    return data


# In[802]:


def get_categories(topic_model, language):
    topics_frame = topic_model.get_topic_info()
    if language == "english":
        category_list = ("entertainment", "sport", "business", "politics", "climate", "medicine", "none")
    else:
        category_list = ("etat", "inland", "international", "kultur", "panorama", "sport", "web", "wirtschaft", "wissenschaft")
        
    number = -1
    values = []
    while(number != len(topics_frame)-1):
        values.append(topic_model.get_topic(number))
        number += 1

    value_list = []
    value_final = []
    for value in values:
        end = 5
        begin = 0
        if type(value) == bool:
            value_list.append("None")
            value_final.append(value_list)
            value_list = []
        else:
            while value[begin] != value[end]:
                value_list.append(value[begin][0])
                begin += 1
            value_final.append(value_list)
            value_list = []
    topics_frame["values"] = value_final
    
    return topics_frame, category_list, value_final


# In[803]:


def word2vecs(topics_frame, category_list, value_final, language):
    google_news_vectors = gensim.downloader.load('word2vec-google-news-300')
    word_vectors = KeyedVectors.load_word2vec_format("C:/Users/svcsc/Desktop/Code/german.model", binary=True)  

    sim = 0.0
    row = 0
    topic = 0
    word = 0
    labeling = []
    label = []
    while(row <= len(topics_frame)-1):
        for value in value_final:
            try:
                cosine = google_news_vectors.similarity(value_final[row][word], category_list[topic])
                if language == "english":
                    cosine = google_news_vectors.similarity(value_final[row][word], category_list[topic])
                if language == "german":
                    cosine = word_vectors.similarity(value_final[row][word], category_list[topic])
            except:
                cosine= 0.0
            if sim < cosine:
                sim = cosine
                cat = category_list[topic]
                print(cat)
            topic += 1
            if topic == 7:
                break
        labeling.append(cat)
        word += 1
        topic = 0
        if word == 5:
                topic = 0
                row += 1
                word = 0
                sim = 0.0
                frequency_distribution = nltk.FreqDist(labeling)
                most_common_element = frequency_distribution.max()
                label.append(most_common_element)
                labeling = []

    topics_frame["label"] = label
    return topics_frame


# In[804]:


def get_label(topic, topics_frame):
    item = (topics_frame.loc[topics_frame['Topic'] == topic]['label']).item()
    return item


# In[805]:


def check_BERTopic(transcript, language):
    transc = pd.read_csv(transcript, index_col=0)
    stops_eng  = set(stopwords.words('english'))
    stops_ger  = set(stopwords.words('german'))
    english_stopwords = "C:/Users/svcsc/Desktop/Code/Stopwörter/pyrouge.txt"
    german_stopword = "C:/Users/svcsc/Desktop/Code/Stopwörter/deutsche_stopwoerter.txt"
    bbc_input_new = "C:/Users/svcsc/Desktop/Code/Stopwörter/bbc_english_ner.txt"
    german_articles_input_new = "C:/Users/svcsc/Desktop/Code/Stopwörter/german_articles_english_ner.txt"

    if language == "english":
        new_words = open(english_stopwords, "r", encoding="utf8").read().split()
        new_stopwords = stops_eng.union(new_words)
        new_words2 = open(bbc_input_new, "r", encoding="utf8").read().lower().split()
        new_stopwords2 = stops_eng.union(new_words2)
        save_train = "C:/Users/svcsc/Desktop/Code/BERTopic/Englisch/data_train.csv"
        save_test = "C:/Users/svcsc/Desktop/Code/BERTopic/Englisch/data_test.csv"
        data_before = "C:/Users/svcsc/Desktop/Code/bbc_preprocessed_complete.csv"
    else:
        new_words = open(german_stopword, "r", encoding="utf8").read().split()
        new_stopwords = stops_ger.union(new_words)
        imput_new = german_articles_input_new
        new_words2 = open(german_articles_input_new, "r", encoding="utf8").read().lower().split()
        new_stopwords2 = stops_ger.union(new_words2)
        save_train = "C:/Users/svcsc/Desktop/Code/BERTopic/Deutsch/data_train.csv"
        save_test = "C:/Users/svcsc/Desktop/Code/BERTopic/Deutsch/data_test.csv"
        data_before = "C:/Users/svcsc/Desktop/Code/german_articles_preprocessed_train.csv"
        
    df = pd.read_csv(data_before, index_col=0)
    train_text, test_text = train_test_split(df, train_size=0.9)
    
    topic_model = make_model(language)
    print("topic_model")
    
    transcript_preprocessed = preprocessText(transc, new_stopwords, new_stopwords2)
    transcript_preprocessed["lemmata"] = transcript_preprocessed.apply(lambda row: cleanLemma(row["lemmata"]),axis = 1)
    transcript_preprocessed.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/process.csv")
    transcript_preprocessed["lemmata"] = transcript_preprocessed["lemmata"].apply(lambda x: ' '.join(x))
    
    topics, probabilities = get_topics_and_probs(topic_model, transcript_preprocessed["lemmata"])
    data =get_topic_prediction(topic_model, transcript_preprocessed)
    topics_frame, category_list, value_final = get_categories(topic_model, language)
    data.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/data.csv")
    topics_frame.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/topics_frame.csv")
    topics_frame = word2vecs(topics_frame, category_list, value_final, language)
      
    data["labels"] = data.apply(lambda row: get_label(int(row["topic_prediction"]),topics_frame), axis=1)
    newData = data[["text", "labels"]]
    return newData


# In[806]:


def check_model():
    model = model_cb.get()
    data_input_sd_de = "C:/Users/svcsc/Desktop/Code/Speaker/deutsch/transcript.csv"
    data_input_sd_en = "C:/Users/svcsc/Desktop/Code/Speaker/englisch/transcript.csv"
    data_input_tt_de = "C:/Users/svcsc/Desktop/Code/TextTiling/deutsch/texttilling.csv.csv"
    data_input_tt_en = "C:/Users/svcsc/Desktop/Code/TextTiling/englisch/texttilling.csv.csv"
    if len(model_cb.get())==0:
        messagebox.showwarning('warning', 'You need to select a model!')
    if language_cb.get() == 0:
                messagebox.showwarning('warning', 'you have to decide a language!')
    else:
        language = language_cb.get()
        model = model_cb.get()
        if model == "snorkel_sd":   
            if language == "de":
                data_output = get_snorkel_german(data_input_sd_de)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Deutsch/Ergebnisse/topics_snorkel_sd.csv")
            else:
                data_output = get_snorkel(data_input_sd_en)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Englisch/Ergebnisse/topics_snorkel_sd.csv")
            output = Label(root, text="Snorkel ready", bg='#B2DFEE')
            output.pack()
        elif model == "snorkel_tt":
            if language == "de":
                data_output = get_snorkel_german(data_input_tt_de)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Deutsch/Ergebnisse/topics_snorkel_tt.csv")
            else:
                data_output = get_snorkel(data_input_tt_en)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Snorkel/Englisch/Ergebnisse/topics_snorkel_tt.csv")
            output = Label(root, text="Snorkel ready", bg='#B2DFEE')
            output.pack()        
        
        elif model == "markov_sd":
            if language == "de":
                data_output = check_markov_german(data_input_sd_de)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Ergebnisse/topics_markov_sd.csv")
            else:
                data_output = check_markov(data_input_sd_en)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/Ergebnisse/topics_markov_sd.csv")
            output = Label(root, text="Markov ready", bg='#B2DFEE')
            output.pack()
        elif model == "markov_tt":
            if language == "de":
                data_output = check_markov_german(data_input_tt_de)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Ergebnisse/topics_markov_tt.csv")
            else:
                data_output = check_markov(data_input_tt_en)
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/Markov/Englisch/Ergebnisse/topics_markov_tt.csv")
            output = Label(root, text="Markov ready", bg='#B2DFEE')
            output.pack()
        
        elif model == "BERTopic_sd":
            if language == "de":
                data_output = check_BERTopic(data_input_sd_de, language= "german")
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/Deutsch/Ergebnisse/topics_BERTopic_sd.csv")
            else:
                data_output = check_BERTopic(data_input_sd_en, language= "english")
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/Englisch/Ergebnisse/topics_BERTopic_sd.csv")
            output = Label(root, text="BERTopic ready", bg='#B2DFEE')
            output.pack()
        elif model == "BERTopic_tt":
            if language == "de":
                data_output = check_BERTopic(data_input_tt_de, language="german")
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/Deutsch/Ergebnisse/topics_BERTopic_tt.csv")
            else:
                data_output = check_BERTopic(data_input_tt_en, language= "english")
                data_output.to_csv("C:/Users/svcsc/Desktop/Code/BERTopic/Englisch/Ergebnisse/topics_BERTopic_tt.csv")
            output = Label(root, text="BERTopic ready", bg='#B2DFEE')
            output.pack()


# In[807]:


##################################################################
# Clear and Table
##################################################################


# In[808]:


def clear_all():
    for widgets in TableMargin.winfo_children():
        widgets.destroy()
    entry.delete(0,END)


# In[809]:


def get_table():
    model = model_cb.get()
    language = language_cb.get()
    if len(model_cb.get())==0:
        messagebox.showwarning('warning', 'You need to select a model!')
    else:
        if model == "markov_sd" and language == "de":
            data = "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Ergebnisse/topics_markov_sd.csv"      
        if model == "markov_sd" and language == "en":
            data = "C:/Users/svcsc/Desktop/Code/Markov/Englisch/Ergebnisse/topics_markov_sd.csv"
        if model == "markov_tt" and language == "de":
            data = "C:/Users/svcsc/Desktop/Code/Markov/Deutsch/Ergebnisse/topics_markov_tt.csv"      
        if model == "markov_tt" and language == "en":
            data = "C:/Users/svcsc/Desktop/Code/Markov/Englisch/Ergebnisse/topics_markov_tt.csv"  
        if model == "snorkel_sd" and language == "de":
            data = "C:/Users/svcsc/Desktop/Code/Snorkel/Deutsch/Ergebnisse/topics_snorkel_sd.csv"
        if model == "snorkel_sd" and language == "en":
            data = "C:/Users/svcsc/Desktop/Code/Snorkel/Englisch/Ergebnisse/topics_snorkel_sd.csv"
        if model == "snorkel_tt" and language == "de":
            data = "C:/Users/svcsc/Desktop/Code/Snorkel/Deutsch/Ergebnisse/topics_snorkel_tt.csv"
        if model == "snorkel_tt" and language == "en":
            data = "C:/Users/svcsc/Desktop/Code/Snorkel/Englisch/Ergebnisse/topics_snorkel_tt.csv"
        if model == "BERTopic_sd" and language == "de":
            data = "C:/Users/svcsc/Desktop/Code/BERTopic/Deutsch/Ergebnisse/topics_BERTopic_sd.csv"
        if model == "BERTopic_sd" and language == "en":
            data = "C:/Users/svcsc/Desktop/Code/BERTopic/Englisch/Ergebnisse/topics_BERTopic_sd.csv"
        if model == "BERTopic_tt" and language == "de":
            data = "C:/Users/svcsc/Desktop/Code/BERTopic/Deutsch/Ergebnisse/topics_BERTopic_tt.csv"
        if model == "BERTopic_tt" and language == "en":
            data = "C:/Users/svcsc/Desktop/Code/BERTopic/Englisch/Ergebnisse/topics_BERTopic_tt.csv"
    TableMargin.pack(side=BOTTOM)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin, columns=("text", "label"), selectmode="extended", yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)

    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)

    tree.heading('text', text="text", anchor=W)
    tree.heading('label', text="label", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=400)
    tree.column('#2', stretch=NO, minwidth=0, width=100)
    tree.pack()

    try:
        with open(data, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                text = row['text']
                label = row['labels']
                tree.insert("", 0, values=(text, label))
    except:
        pass


# In[810]:


root.geometry("600x600")    
file_menu = Menu(menubar, background='white', foreground='black', activebackground='#FFC125', activeforeground='black', tearoff=False)
file_menu.add_command(label='Exit',command=root.destroy,)
file_menu.add_command(label='New',command=clear_all)
menubar.add_cascade(label="File", menu=file_menu,underline=0)

user_input = StringVar()
selected_language = StringVar()
selected_model = StringVar()
TableMargin = Frame(root, width=500)

url = Label(root, text = "YouTube-URL", bg="#F6F6F6", font=LARGE_FONT)
url.pack(padx=10)
url.place(relx = 0.1, rely = 0.1)

entry = Entry(root, textvariable = user_input, font=LARGE_FONT, width=40)
entry.pack(padx=10)
entry.place(relx = 0.3, rely = 0.1)

language = Label(root, text = "Language", bg='#F6F6F6', font=LARGE_FONT)
language.pack()
language.place(relx = 0.1, rely = 0.15)

language_cb = ttk.Combobox(root, textvariable=selected_language, width=15)
language_cb['values'] = ["en", "de"]
language_cb['state'] = 'readonly'
language_cb.pack(pady=10)
language_cb.place(relx = 0.3, rely = 0.15)

language = Label(root, text = "Modell", bg='#F6F6F6', font=LARGE_FONT)
language.pack()
language.place(relx = 0.1, rely = 0.2)

model_cb = ttk.Combobox(root, textvariable=selected_model, width=15)
model_cb['values'] = ["markov_sd", "markov_tt", "snorkel_sd", "snorkel_tt", "BERTopic_sd", "BERTopic_tt"]
model_cb['state'] = 'readonly'
model_cb.pack(pady=10)
model_cb.place(relx = 0.3, rely = 0.2)

###############################################################################################
button = Button(root, text = "download",width=15,command = get_check, bg="#CF1820", font=LARGE_FONT)
button.pack()
button.place(relx = 0.1, rely = 0.3)

button = Button(root, text = "speaker diarization",width=15,command = cut_all_files, bg="#AF368C", font=LARGE_FONT)
button.pack()
button.place(relx = 0.1, rely = 0.35)

button = Button(root, text = "texttiling",width=15,command = check_for_transcript_texttiling, bg="#EC6525", font=LARGE_FONT)
button.pack()
button.place(relx = 0.1, rely = 0.4)

################################################################################################
button = Button(root, text = "model",width=15,command = check_model, bg="#1BA48A", font=LARGE_FONT)
button.pack()
button.place(relx = 0.6, rely = 0.3)


button = Button(root, text = "table",width=15,command = get_table, bg="#1BA48A", font=LARGE_FONT)
button.pack()
button.place(relx = 0.6, rely = 0.35)
##############################################################################################


root.mainloop()


# In[ ]:





# In[ ]:




