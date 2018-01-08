

import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from steeming import steemer 
import pickle
import numpy as np
#from preprocessing import removePunctuation

from preprocessing import removePunctuations,removeStopWords,SentencePositionScore,SentenceLengthScore,TitleScore
#tokenizing - word tokenizers

def getdgist():
    fo = open("news.txt","r+")
    example_text=fo.read()
    sentences = sent_tokenize(example_text)

    fo = open("news_title.txt","r+")
    title=fo.read()

    sentencearray=[]
    sent_tfidf=[]
    sent_title=[]
    sent_position=[]
    sent_len=[]
    test_data=[]

    example_array=word_tokenize(example_text)
    print("Length of Original text ->",len(example_array))
    print()
    
    for each_sent in sentences:
        each_sent=removePunctuations(each_sent.lower())
        sent_words=word_tokenize(each_sent)
        sent_words=removeStopWords(sent_words)
        sent_words=steemer(sent_words)
        sentence=' '.join(sent_words)
        sentencearray.append(sentence)
                
 
    tf_pickle=open("tf.pickle","rb")
    vect=pickle.load(tf_pickle)

    test_dtm=vect.transform(sentencearray)

    tfidf_pickle=open("tfidf.pickle","rb")          
    tfidf=pickle.load(tfidf_pickle)

    tf_idf_matrix=tfidf.transform(test_dtm)

    TF=tf_idf_matrix.astype(np.float32) 

    Tf_Idf=TF.toarray()


    for index,each_TfIdf in enumerate(Tf_Idf):
        l=len(word_tokenize(sentencearray[index]))
        summ=0
        if l!=0:
            for i in each_TfIdf:
                summ+=i
                summ=summ/l

    sent_tfidf.append(summ)

    Title=removePunctuations(title)
    titleWords=word_tokenize(Title)
    titleWords=removeStopWords(titleWords)
    titleWords=steemer(titleWords)


    print("Length of titlewords ->",len(titleWords))
    print(titleWords)
    

    for each_sent in sentencearray:
            sentenceWords=word_tokenize(each_sent)
            sent_title.append(TitleScore(titleWords,sentenceWords))


    length=len(sentencearray)
    for index,each_sent in enumerate(sentencearray):
        sent_position.append(SentencePositionScore(index+1,length))


    max_l=[]
    for each_sent in sentencearray:
        max_l.append(len(word_tokenize(each_sent)))
        
        max_length=max(max_l)


    for i in range(len(sentencearray)):
        sent_len.append(SentenceLengthScore(sentencearray[i],max_length))



    for i,j,k,l in zip(sent_tfidf,sent_title,sent_position,sent_len):
        test_data.append([i,j,k,l])


    classifier_pickle=open("clf.pickle","rb")
    clf=pickle.load(classifier_pickle)
   # print("sentencearray ->",len(sentencearray))
    count=0
    for i in sentence: 
        count=count+len(i)
    print("Length of Summary ->",count)   
    print(sentencearray)
    
    prediction=clf.predict(test_data)
    print(prediction)       


getdgist()