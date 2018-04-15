 # -*- coding: utf-8 -*-
import textract
from txt_word import TxtWord
import networkx as nx

#choose your file here
file=textract.process('d:/testData/quality_iss.txt')
sentences=file.decode('utf8').split('\n')
tword=TxtWord(sentences)
tword.stop_words.append('没有')
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
tword.plot_word_histogram(freq,show=50)

#in order to create a good graph, I combined some sentences, your can remove these lines
lines=[]
c=16
for i in range(int(len(sentences)/c)):
    line=''
    for j in range(c):
        line=line+sentences[c*i+j]
    lines.append(line)

co_occ=tword.co_occurrences(lines,keywords)
wgraph=tword.co_occurrence_graph(freq,co_occ,del_isolated=True)
tword.plot_graph(wgraph)

