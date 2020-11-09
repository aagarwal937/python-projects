#Requried imports
import os
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import numpy as np

cik_List_File = pd.read_excel('cik_List.xlsx')

#Section 1.1: Positive score, negative score, polarity score


# Tokenizer
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)



# Calculating Average sentence length 
# It will calculated using formula --- Average Sentence Length = the number of words / the number of sentences
     
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length= average_sent
    
    return round(average_sent_length)

# Calculating percentage of complex word 
# It is calculated using Percentage of Complex words = the number of complex words / the number of words 

def percentage_complex_word(text):
    tokens = tokenizer(text)
    complexWord = 0
    complex_word_percentage = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    if len(tokens) != 0:
        complex_word_percentage = complexWord/len(tokens)
    
    return complex_word_percentage

# calculating Fog Index 
# Fog index is calculated using -- Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)

def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return fogIndex

# Counting complex words
def complex_word_count(text):
    tokens = tokenizer(text)
    complexWord = 0
    
    for word in tokens:
        vowels=0
        if word.endswith(('es','ed')):
            pass
        else:
            for w in word:
                if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                    vowels += 1
            if(vowels > 2):
                complexWord += 1
    return complexWord

#Counting total words

def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

# calculating uncertainty_score
with open(uncertainty_dictionaryFile ,'r') as uncertain_dict:
    uncertainDict=uncertain_dict.read().lower()
uncertainDictionary = uncertainDict.split('\n')

def uncertainty_score(text):
    uncertainWordnum =0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in uncertainDictionary:
            uncertainWordnum +=1
    sumUncertainityScore = uncertainWordnum 
    
    return sumUncertainityScore

# calculating constraining score
with open(constraining_dictionaryFile ,'r') as constraining_dict:
    constrainDict=constraining_dict.read().lower()
constrainDictionary = constrainDict.split('\n')

def constraining_score(text):
    constrainWordnum =0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in constrainDictionary:
            constrainWordnum +=1
    sumConstrainScore = constrainWordnum 
    
    return sumConstrainScore

# Calculating positive word proportion

def positive_word_prop(positiveScore,wordcount):
    positive_word_proportion = 0
    if wordcount !=0:
        positive_word_proportion = positiveScore / wordcount
        
    return positive_word_proportion

# Calculating negative word proportion

def negative_word_prop(negativeScore,wordcount):
    negative_word_proportion = 0
    if wordcount !=0:
        negative_word_proportion = negativeScore / wordcount
        
    return negative_word_proportion

# Calculating uncertain word proportion

def uncertain_word_prop(uncertainScore,wordcount):
    uncertain_word_proportion = 0
    if wordcount !=0:
        uncertain_word_proportion = uncertainScore / wordcount
        
    return uncertain_word_proportion

# Calculating constraining word proportion

def constraining_word_prop(constrainingScore,wordcount):
    constraining_word_proportion = 0
    if wordcount !=0:
        constraining_word_proportion = constrainingScore / wordcount
        
    return constraining_word_proportion

# calculating Constraining words for whole report

def constrain_word_whole(mdaText,qqdmrText,rfText):
    wholeDoc = mdaText + qqdmrText + rfText
    constrainWordnumWhole =0
    rawToken = tokenizer(wholeDoc)
    for word in rawToken:
        if word in constrainDictionary:
            constrainWordnumWhole +=1
    sumConstrainScoreWhole = constrainWordnumWhole 
    
    return sumConstrainScoreWhole

inputDirectory = 'D:/data science/Blackcoffer project/test'
masterFile = 'D:/data science/Blackcoffer project/cik_list1.csv'
dataList = rawdata_extract( inputDirectory , masterFile )
df = pd.DataFrame(dataList)

df['mda_positive_score'] = df.mda_extract.apply(positive_score)
df['mda_negative_score'] = df.mda_extract.apply(negative_word)
df['mda_polarity_score'] = np.vectorize(polarity_score)(df['mda_positive_score'],df['mda_negative_score'])
df['mda_average_sentence_length'] = df.mda_extract.apply(average_sentence_length)
df['mda_percentage_of_complex_words'] = df.mda_extract.apply(percentage_complex_word)
df['mda_fog_index'] = np.vectorize(fog_index)(df['mda_average_sentence_length'],df['mda_percentage_of_complex_words'])
df['mda_complex_word_count']= df.mda_extract.apply(complex_word_count)
df['mda_word_count'] = df.mda_extract.apply(total_word_count)
df['mda_uncertainty_score']=df.mda_extract.apply(uncertainty_score)
df['mda_constraining_score'] = df.mda_extract.apply(constraining_score)
df['mda_positive_word_proportion'] = np.vectorize(positive_word_prop)(df['mda_positive_score'],df['mda_word_count'])
df['mda_negative_word_proportion'] = np.vectorize(negative_word_prop)(df['mda_negative_score'],df['mda_word_count'])
df['mda_uncertainty_word_proportion'] = np.vectorize(uncertain_word_prop)(df['mda_uncertainty_score'],df['mda_word_count'])
df['mda_constraining_word_proportion'] = np.vectorize(constraining_word_prop)(df['mda_constraining_score'],df['mda_word_count'])

df['qqdmr_positive_score'] = df.qqd_extract.apply(positive_score)
df['qqdmr_negative_score'] = df.qqd_extract.apply(negative_word)
df['qqdmr_polarity_score'] = np.vectorize(polarity_score)(df['qqdmr_positive_score'],df['qqdmr_negative_score'])
df['qqdmr_average_sentence_length'] = df.qqd_extract.apply(average_sentence_length)
df['qqdmr_percentage_of_complex_words'] = df.qqd_extract.apply(percentage_complex_word)
df['qqdmr_fog_index'] = np.vectorize(fog_index)(df['qqdmr_average_sentence_length'],df['qqdmr_percentage_of_complex_words'])
df['qqdmr_complex_word_count']= df.qqd_extract.apply(complex_word_count)
df['qqdmr_word_count'] = df.qqd_extract.apply(total_word_count)
df['qqdmr_uncertainty_score']=df.qqd_extract.apply(uncertainty_score)
df['qqdmr_constraining_score'] = df.qqd_extract.apply(constraining_score)
df['qqdmr_positive_word_proportion'] = np.vectorize(positive_word_prop)(df['qqdmr_positive_score'],df['qqdmr_word_count'])
df['qqdmr_negative_word_proportion'] = np.vectorize(negative_word_prop)(df['qqdmr_negative_score'],df['qqdmr_word_count'])
df['qqdmr_uncertainty_word_proportion'] = np.vectorize(uncertain_word_prop)(df['qqdmr_uncertainty_score'],df['qqdmr_word_count'])
df['qqdmr_constraining_word_proportion'] = np.vectorize(constraining_word_prop)(df['qqdmr_constraining_score'],df['qqdmr_word_count'])

df['rf_positive_score'] = df.riskfactor_extract.apply(positive_score)
df['rf_negative_score'] = df.riskfactor_extract.apply(negative_word)
df['rf_polarity_score'] = np.vectorize(polarity_score)(df['rf_positive_score'],df['rf_negative_score'])
df['rf_average_sentence_length'] = df.riskfactor_extract.apply(average_sentence_length)
df['rf_percentage_of_complex_words'] = df.riskfactor_extract.apply(percentage_complex_word)
df['rf_fog_index'] = np.vectorize(fog_index)(df['rf_average_sentence_length'],df['rf_percentage_of_complex_words'])
df['rf_complex_word_count']= df.riskfactor_extract.apply(complex_word_count)
df['rf_word_count'] = df.riskfactor_extract.apply(total_word_count)
df['rf_uncertainty_score']=df.riskfactor_extract.apply(uncertainty_score)
df['rf_constraining_score'] = df.riskfactor_extract.apply(constraining_score)
df['rf_positive_word_proportion'] = np.vectorize(positive_word_prop)(df['rf_positive_score'],df['rf_word_count'])
df['rf_negative_word_proportion'] = np.vectorize(negative_word_prop)(df['rf_negative_score'],df['rf_word_count'])
df['rf_uncertainty_word_proportion'] = np.vectorize(uncertain_word_prop)(df['rf_uncertainty_score'],df['rf_word_count'])
df['rf_constraining_word_proportion'] = np.vectorize(constraining_word_prop)(df['rf_constraining_score'],df['rf_word_count'])

df['constraining_words_whole_report'] = np.vectorize(constrain_word_whole)(df['mda_extract'],df['qqd_extract'],df['riskfactor_extract'])

print(df.shape)

inputTextCol = ['mda_extract','qqd_extract','riskfactor_extract']
finalOutput = df.drop(inputTextCol,1)

print(finalOutput.head(150))