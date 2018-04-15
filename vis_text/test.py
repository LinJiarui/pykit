 # -*- coding: utf-8 -*-
import textract
from txt_word import TxtWord
import networkx as nx
import os

#choose your file here
file=textract.process('g:/Develop/pykit/data/example.txt')
sentences=file.decode('utf8').split('\n')
tword=TxtWord(sentences)
tword.stop_words.append('没有')#append new stop words
tags=tword.Tfidf_words(allow_pos=())
print(tags)
keywords=tword.rank_words()
print(keywords)
keyphrases=tword.rank_phrases()
print(keyphrases)

words=tword.seg_all()
fwc=tword.gen_cloud(words)
tword.plot_cloud(fwc)

freq=tword.word_freq(words)
tword.summarize_word_freq(freq)
tword.plot_word_histogram(freq,show=25)

#in order to create a good graph, I combined some sentences, your can remove these lines
lines=[]
c=1
for i in range(int(len(sentences)/c)):
    line=''
    for j in range(c):
        line=line+sentences[c*i+j]
    lines.append(line)

co_occ=tword.co_occurrences(sentences,keywords)
wgraph=tword.co_occurrence_graph(freq,co_occ,del_isolated=True,node_cutoff=16,edge_cutoff=4)
tword.plot_graph(wgraph)

#w2v_model=tword.get_wv_model()
#print(w2v_model.wv)

keywords=tword.rank_words(count=50)
vec_list=tword.get_word_vec(keywords)
tword.plot_dendrogram(keywords,vec_list)

clusters=tword.kmeans_cluster(vec_list)
tword.plot_cluster(clusters,keywords,vec_list)
