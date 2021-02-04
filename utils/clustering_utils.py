from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import networkx as nx


class Cluster:
    def __init__(self, texts, center, label, silhouette):
        self.texts = texts
        self.center = center
        self.label = label
        self.silhouette = silhouette

    def __str__(self):
        return "("+self.label+") "+str(self.texts)

    def get_texts(self):
        return self.texts


class Clustering:
    def __init__(self, data, vectorizer, model, labels,
                 clusters, inertia, silhouette, X_embedded):
        self.data = data
        self.vectorizer = vectorizer
        self.model = model
        self.labels = labels
        self.clusters = clusters
        self.inertia = inertia
        self.silhouette = silhouette
        self.X_embedded = X_embedded

    def print_meta(self):
        print("data : ")
        print(self.data)
        print("model")
        print(self.model)
        print("labels")
        print(self.labels)
        print("clusters")
        print(self.clusters)
        print("inertia")
        print(self.inertia)
        print("silhouette score")
        print(self.silhouette)

    def plot(self):
        """
            plot clustered documents
        """
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(frameon=False)
        plt.setp(ax, xticks=(), yticks=())
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                            wspace=0.0, hspace=0.0)
        plt.scatter(self.X_embedded[:, 0], self.X_embedded[:, 1],
                    c=self.labels, marker="o")

        plt.scatter(self.model.cluster_centers_[:, 0],
                    self.model.cluster_centers_[:, 1],
                    marker='+',
                    color='black',
                    s=200)

    def print(self):
        lines = []
        # clust_size = lambda clust : len(clust.texts)
        # sort based on cluster length
        self.clusters.sort(key=lambda x: len(x.texts))
        for idx, cluster in enumerate(self.clusters):
            print("# cluster="+str(idx)+"\t #docs="
                  + str(len(cluster.texts))+"\n")
            txtidx = 0
            for text in cluster.texts:
                print("@"+str(txtidx)+"   "+text+"\n")
                txtidx += 1

    def print_descr(self):
        """
                print descriptive info about cluster size
        """
        print("Cluster\tSize\tCenter\t")
        for idx, cluster in enumerate(self.clusters):
            print(str(idx)+"\t")
            print(str(len(cluster.texts))+"\t")
            print(cluster.center)

    def get_top_keywords(self, n_terms):
        """
            return n_terms top keywords for each cluster
        """
        clusters = [x for x in range(0, len(self.clusters))]  # clusters indexs
        df = pd.DataFrame(self.data.todense()).groupby(clusters).mean()
        labels = self.vectorizer.get_feature_names()

        for i, r in df.iterrows():
            print('\nCluster {}'.format(i))
            print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))

    def save(self, fileout):
        lines = []
        self.clusters.sort(key=lambda x: len(x.texts))
        for idx, cluster in enumerate(self.clusters):
            lines.append("#### Cluster "+str(idx)+"\t #"
                         + str(len(cluster.texts))+" docs ####\n")
            txtidx = 0
            for doc in cluster.texts:
                lines.append("@"+str(txtidx)+"  "+doc+"\n\n")
                txtidx += 1
        with open(fileout, "w") as outfile:
            outfile.writelines(lines)

    def save_edus(self, corpus, fileout):
        """
         todo : add corpus to object attributes ?
        """
        lines = []
        self.clusters.sort(key=lambda x: len(x.texts))
        for idx, cluster in enumerate(self.clusters):
            lines.append("#### Cluster "+str(idx)+"\t #"
                         + str(len(cluster.texts))+" docs ####\n")
            # recover doc id
            for (text, docid) in cluster.texts:
                lines.append('@ '+str(docid)+"\n")
                # loop over edus and create line
                for eduid, edu in enumerate(corpus.loc[docid, "edustxt"]):
                    lines.append('@ '+str(docid)+"_"+str(eduid+1)+" "+edu+"\n")
        with open(fileout, "w") as outfile:
            outfile.writelines(lines)

    def save_graphs(self, corpus, dirout):
        """
        dirout est le nom du dossier où ssauver les images des graphs
        ensuite 1 dossier par cluster, et iddoc.png l'image du graph iddoc
        est sauvée dans le dossier correspondant à son cluster
        """
        self.clusters.sort(key=lambda x: len(x.texts))
        for idx, cluster in enumerate(self.clusters):
            # create dir to put images
            clustdir = dirout+"cluster_"+str(idx)
            print("Populating "+clustdir+" with graph imgs....")
            os.mkdir(clustdir)
            for (text, docid) in cluster.texts:
                # recover graph
                graph = corpus.loc[docid, "graph"]
                # save to file as png img
                graphimg = nx.nx_agraph.to_agraph(graph)  # interface  pygviz
                graphimg.layout(prog="dot")               # add pos to nodes
                graphimg.draw(clustdir+"/"+docid+".png")  # draw to file


# Glob functions

def best_clusterings(clusterings):
    """
        useful to find best N nb of clusters for a clustering
        remove clusterings where 1 cluster silhouette is smaller
        than avg silhouette of clusters
    """
    best_clusterings = {}
    for idx, clustering in clusterings.items():
        append = True
        for cluster in clustering.clusters:
            if max(cluster.silhouette) < clustering.silhouette:
                append = False
                break
        if append:
            best_clusterings[idx] = clustering
    return best_clusterings


def kmeans(documents, corpus, nbclusters):
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    X = vectorizer.fit_transform(documents)
    saveX = copy.deepcopy(X)
    model = KMeans(n_clusters=nbclusters,
                   init='k-means++',
                   max_iter=100,
                   n_init=1,
                   random_state=42)
    model.fit(X)
    # order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    # terms = vectorizer.get_feature_names()
    silhouette_sc = silhouette_score(X, model.labels_)
    # print("silhouette score avg : " + str(silhouette_sc))
    silhouette_sam = silhouette_samples(X, model.labels_)
    clusters = [[] for i in range(nbclusters)]
    labels = model.labels_

    # reduce data dimension to plot in 2D
    X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
    X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)

    for idx, doc in enumerate(documents):
        docid = corpus[corpus['text'] == doc].index.tolist()
        docid = docid[0]
        X = vectorizer.transform([documents[idx]])
        predicted = model.predict(X)
        predicted = int(predicted[0])
        clusters[predicted].append((doc, docid))

    clusterslist = []
    # create clusters objects (could be done in upper loop)
    for (label, center, texts) in zip([x for x in range(len(clusters))],
                                      model.cluster_centers_, clusters):
        silhouette = None
        silhouette = silhouette_sam[labels == label]
        silhouette.sort()
        size_cluster_i = silhouette.shape[0]
        # create cluster object and add it to clusters list
        c = Cluster(texts, center, label, silhouette)
        clusterslist.append(c)

    return Clustering(saveX, vectorizer, model, model.labels_,
                      clusterslist, model.inertia_, silhouette_sc, X_embedded)
