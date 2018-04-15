#-*- encoding:utf-8 -*-
import os
import re
import codecs
from collections import Counter
import operator

import numpy as np
import pandas as pd

import jieba
import jieba.posseg as psg
import jieba.analyse as als
import gensim

from scipy.misc import imread
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS

from wordcloud import WordCloud,ImageColorGenerator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import networkx as nx

def get_default_img_mask():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, 'cloud.jpg')

def get_default_stop_words_file():
    d = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(d, 'stopwords.txt')
sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
allow_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']

class TxtWord:
    def __init__(self,sentences,stop_words_file=None,font='SimHei',font_size=10,figsize=(8,8)):
        self.sentences=sentences
        self.stop_words=[]
        #todo: add update function for the following parameters
        self._font=font
        self._font_size=font_size
        self._figsize=figsize
        
        self._w2v_min_count=2
        
        self.__text=''
        self.__words={}
        self.__sen_words=[]
        self.__w2v_model=None
        
        self.stop_words_file = get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.stop_words_file = stop_words_file
        
        for word in codecs.open(self.stop_words_file, 'r', 'utf-8', 'ignore'):
            self.stop_words.append(word.strip())
            
        self._cf=FontProperties(font)
        matplotlib.rc('font',family=font)
        matplotlib.rcParams['figure.figsize']=figsize
        matplotlib.rcParams['font.size']=font_size
    
    def _inner_cleanup(self):
        self.__text=''
        self.__words={}
        self.__sen_words=[]
        self.__w2v_model=None
    
    def _inner_init(self,init_txt_only=False):
        if len(self.__text)==0:
            self.__text='\n'.join(self.sentences)
        
        if len(self.__sen_words)==0 and not init_txt_only:
            for sent in self.sentences:
                self.__sen_words.append(list(psg.cut(sent)))
        
        if len(self.__sen_words)>0 and not init_txt_only and self.__w2v_model==None:
            sen_words=[[w.word for w in sen] for sen in self.__sen_words]
            self.__w2v_model=gensim.models.Word2Vec(sen_words, min_count=self._w2v_min_count)
    
    def regen_sentences(self):
        self._inner_init(init_txt_only=True)
        res = [self.__text]
        
        for sep in sentence_delimiters:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        self.sentences=res
        
        self._inner_init()
        
        
    def Tfidf_words(self,count=36,allow_pos=('ns','n','vn','v')):
        self._inner_init()
        
        return als.extract_tags(self.__text,topK=count,allowPOS=allow_pos,withWeight=False)
    
    def rank_words(self,count=36,allow_pos=('ns','n','vn','v')):
        self._inner_init()
        
        return als.textrank(self.__text,topK=count,allowPOS=allow_pos,withWeight=False)
    
    def rank_phrases(self, countK=36, min_occur=2): 
        self._inner_init()
        
        keywords=self.rank_words(countK)
        keyphrases = set()
        for sentence in self.__sen_words:
            one = []
            for word in sentence:
                if word in keywords:
                    one.append(word)
                else:
                    if len(one) >  1:
                        keyphrases.add(''.join(one))
                    if len(one) == 0:
                        continue
                    else:
                        one = []
            if len(one) >  1:
                keyphrases.add(''.join(one))

        return [phrase for phrase in keyphrases 
                if self.__text.count(phrase) >= min_occur_num]
            
    def seg_all(self,allow_pos=allow_tags):
        """Return words of all sentences as a large list"""
        self._inner_init()
        result=[]
        for sent in self.__sen_words:
            for w in sent:
                if len(w.word)>1 and (not w.word in self.stop_words) and (allow_pos==None, len(allow_pos)==0 or w.flag in allow_pos):
                    result.append(w.word)
        return result
        
    def seg_sentences(self,allow_pos=('ns','n','vn','v')):
        """Return words of each sentence as a nested list"""
        self._inner_init()
        
        result=[]
        for sent in self.__sen_words:
            sarr=[]
            for w in sent:
                if len(w.word)>1 and (not w.word in self.stop_words) and (allow_pos==None, len(allow_pos)==0 or w.flag in allow_pos):
                    sarr.append(w.word)
            if len(sarr)>0:
                result.append(sarr)
        return result
    
    def gen_cloud(self,words,max_words=250,mask_img='',recolor=False):
        seg_space=' '.join(words)
        #wordcloud默认不支持中文，这里的font_path需要指向中文字体，不然得到的词云全是乱码
        #在初始化WordCloud时增加collocations=False信息，以避免文字重复
        fwc=None

        imgmask=get_default_img_mask()
        if len(mask_img)>0:
            imgmask=mask_img
        
        if len(imgmask)==0:
            fwc=WordCloud(font_path='msyh.ttc',
                max_words=max_words,
                background_color='white',
                max_font_size=160,
                font_step=1,
                collocations=False).generate(seg_space)
            return fwc
        else:
            alice_color=imread(imgmask)
            fwc=WordCloud(font_path='msyh.ttc',
                max_words=max_words,
                background_color='white',
                mask=alice_color,
                max_font_size=160,
                font_step=1,
                collocations=False).generate(seg_space)
        
            if recolor:
                imagecolor=ImageColorGenerator(alice_color)
                return fwc.recolor(imagecolor)
            
            return fwc
            
    def plot_cloud(self,fwc):
        plt.imshow(fwc)
        plt.axis('off')
        plt.show()
    
    def _all_pairs(self,items):
        """Make all unique pairs (order doesn't matter)"""
        pairs=[]
        nitems=len(items)
        for i,wi in enumerate(items):
            for j in range(i+1,nitems):
                pairs.append((wi,items[j]))
        
        return pairs
    
    def word_freq(self,words):
        """Return a dictionary of word frequencies for the given text. Input text should be given as an iterable of strings."""
        
        freqs={}
        for word in words:
            freqs[word]=freqs.get(word,0)+1
        
        sorted_x=sorted(freqs.items(),key=operator.itemgetter(1),reverse=False)
        
        return sorted_x
    
    def _print_vk(self,lst):
        """Print a list of value/key pairs nicely formatted in key/value order."""

        # Find the longest key: remember, the list has value/key paris, so the key
        # is element [1], not [0]
        longest_key = max([len(word) for word, count in lst])
        # Make a format string out of it
        fmt = '{}'+str(longest_key)+'s -> {}'
        # Do actual printing
        for k,v in lst:
            print( fmt.format(k,v))
    
    def summarize_word_freq(self,freqs,n=10):
        """Print a simple summary of a word frequencies dictionary.
        freqs : dict or list
          Word frequencies, represented either as a dict of word->count, or as a
          list of count->word pairs.
        n : int
          The number of least/most frequent words to print."""
        
        print('Number of unique words: {}'.format(len(freqs)))
        print()
        print('{} least frequent words:'.format(n))
        self._print_vk(freqs[:n])
        print()
        print('{} most frequent words:'.format(n))
        self._print_vk(freqs[-n:])
        
    def plot_word_histogram(self,freqs,show=10,title=None):
        """Plot a histogram of word frequencies, limited to the top `show` ones."""
        
        # Don't show the tail
        if isinstance(show, int):
            # interpret as number of words to show in histogram
            show_f = freqs[-show:]
        else:
            # interpret as a fraction
            start = -int(round(show*len(freqs)))
            show_f = freqs[start:]

        # Now, extract words and counts, plot
        n_words = len(show_f)
        ind = np.arange(n_words)
        words = [i[0] for i in show_f]
        counts = [i[1] for i in show_f]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if n_words<=25:
            # Only show bars and x labels for small histograms, they don't make sense otherwise
            ax.bar(ind, counts)
            ax.set_xticks(ind)
            ax.set_xticklabels(words, rotation=45,fontproperties = self._cf)
            fig.subplots_adjust(bottom=0.25)
        else:
            # For larger ones, do a step plot
            ax.step(ind, counts)

        # If it spans more than two decades, use a log scale
        if float(max(counts))/min(counts) > 100:
            ax.set_yscale('log')

        if title:
            ax.set_title(title)
        
        plt.show()
        return ax
    
    def _rescale_arr(self,arr,amin,amax):
        """Rescale an array to a new range.
        Return a new array whose range of values is (amin,amax).
        arr : array-like
        amin : float
          new minimum value
        amax : float
          new maximum value
        """
        
        if len(arr)==0:
            return arr
        
        # old bounds
        m = arr.min()
        M = arr.max()
        # scale/offset
        s = float(amax-amin)/(M-m)
        d = amin - s*m

        # Apply clip before returning to cut off possible overflows outside the
        # intended range due to roundoff error, so that we can absolutely guarantee
        # that on output, there are no values > amax or < amin.
        return np.clip(s*arr+d,amin,amax)
    
    def co_occurrences(self,lines, words):
        """Return histogram of co-occurrences of words in a list of lines.
        lines : list
          A list of strings considered as 'sentences' to search for co-occurrences.
        words : list
          A list of words from which all unordered pairs will be constructed and searched for co-occurrences.
        """
        wpairs = self._all_pairs(words)
        
        # Now build histogram of co-occurrences
        co_occur = {}
        for w1, w2 in wpairs:
            rx = re.compile('%s .*%s|%s .*%s' % (w1, w2, w2, w1))
            co_occur[w1, w2] = sum([1 for line in lines if rx.search(line)])
        
        return co_occur
    
    def co_occurrence_graph(self,freqs, co_occur,node_cutoff=50,edge_cutoff=0, del_isolated=False):
        """Convert a word histogram with co-occurrences to a weighted graph.
        Edges are only added if the count is above cutoff.
        """
        g = nx.Graph()
        for word, count in freqs:
            if count>node_cutoff:
                g.add_node(word, count=count)
        for (w1, w2), count in co_occur.items():
            if count<=edge_cutoff:
                continue
            g.add_edge(w1, w2, weight=count)
        
        if del_isolated:
            g.remove_nodes_from(list(nx.isolates(g)))
        return g
    
    def plot_graph(self,wgraph, edge_label=False,pos=None, fig=None):
        """Conveniently summarize graph visually"""
        # Plot nodes with size according to count
        sizes = []
        degrees = []
        for n, d in wgraph.nodes(data=True):
            try:
                sizes.append(d['count'])
            except:
                sizes.append(0)
            degrees.append(wgraph.degree(n))
        sizes = self._rescale_arr(np.array(sizes, dtype=float), 100, 1000)
        #degrees=rescale_arr(np.array(degrees, dtype=float), 96, 255)
       
        # Compute layout and label edges according to weight
        pos = nx.spring_layout(wgraph) if pos is None else pos
        labels = {}
        width = []
        for n1, n2, d in wgraph.edges(data=True):
            w = d['weight']
            labels[n1, n2] = w
            width.append(w)
    
        # remap width to 1-wmax
        wmax = 20
        width = self._rescale_arr(np.array(width, dtype=float), 1, wmax)
        
        # Create figure
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.add_subplot(111)
        fig.subplots_adjust(0,0,1)
        nx.draw_networkx_nodes(wgraph, pos, node_size=sizes, node_color=degrees,alpha=0.4,font_family=self._font,fontproperties = self._cf)
        nx.draw_networkx_labels(wgraph, pos, font_size=24, font_weight='bold',font_family=self._font, fontproperties = self._cf)
        nx.draw_networkx_edges(wgraph, pos, width=width, edge_color=width,edge_cmap=plt.cm.cool)#plt.cm.Blues)
        if edge_label:
            nx.draw_networkx_edge_labels(wgraph, pos, edge_labels=labels, font_size=18,font_family=self._font, fontproperties = self._cf)
        ax.set_title('Node color:degree, size:count, edge: co-occurrence count',fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
   
    def summarize_centrality(self,wgraph):
        c = wgraph.items()
        c.sort(key=lambda x:x[1], reverse=True)
        print('\nGraph centrality')
        for node, cent in c:
            print("%15s: %.3g" % (node, cent))
    
    def get_wv_model(self,min_count=2):
        if self._w2v_min_count!=min_count:
            self._w2v_min_count=min_count
            self._inner_cleanup()
            self._inner_init()
        
        return self.__w2v_model
    
    def train_wv_model(self,sen_words,min_count=2):
        return gensim.models.Word2Vec(sen_words, min_count=min_count)
    
    def get_word_vec(self,words):
        self._inner_init()
        result=[]
        for w in words:
            result.append(self.__w2v_model[w])
        
        return result
    
    def plot_dendrogram(self,keywords,vec_list):
        #based on https://stackoverflow.com/questions/41462711/python-calculate-hierarchical-clustering-of-word2vec-vectors-and-plot-the-resu
        
        l = linkage(vec_list, method='ward', metric='euclidean')
        
        # calculate full dendrogram
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.ylabel('distance')
        plt.xlabel('word')
        dendrogram(
            l,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=self._font_size,  # font size for the x axis labels
            #orientation='left',
            leaf_label_func=lambda v: str(keywords[v])
        )
        
        plt.show()
        
    def kmeans_cluster(self,vec_list,num_clusters=3):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=num_clusters)
        km.fit(vec_list)
        clusters = km.labels_.tolist()
        return clusters

    def _get_cmap(self,n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
    
    def plot_cluster(self,clusters,keywords,vec_list):
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.metrics.pairwise import euclidean_distances
        dist = 1 - cosine_similarity(vec_list)
        
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]
        
        #create data frame that has the result of the MDS plus the cluster numbers and keywords
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, keyword=keywords))
        #group by cluster
        groups = df.groupby('label')
        cmp=plt.cm.get_cmap('spring', len(clusters))
        colors=[cmp(i) for i in range(len(clusters))]
        
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.scatter(group.x, group.y,color=colors[name],label=name)
        #add label in x,y position with the label as the keyword
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['keyword'])
        
        ax.legend(numpoints=1)  #show legend with only 1 point
        plt.show() #show the plot    