import pandas as pd
import string
from preprocessing_utils import (get_root, get_annot, get_docname, get_json,
                                 get_hdrs, get_edus, get_words,
                                 get_edustxt, get_rels, get_graph,
                                 draw_graph, rebuild_txt, get_rels_in_doc,
                                 get_sent_idx)
from clustering_utils import best_clusterings, kmeans
from nltk import sent_tokenize


class Corpus:
    def __init__(self, corpus=None, files=None):
        # ToDo : demander à thomas comment gérer joliement ces arguments
        # case construction is made from already constructed dtf
        if corpus is not None:
            self.corpus = corpus
        # case construction is made from files
        if files is not None:
            # construct and load dtf corpus
            self.corpus = load_dtf(files)

        # load other parameters
        ids = list(self.corpus.index.values)
        texts = [self.corpus.loc[idd, "txt"] for idd in ids]  # pds dataframe input
        self.documents = [(id, text) for (id, text) in zip(ids, texts)]
        self.clusters = []
        self.model = None
        self.cluster_inertia = None

    def save_reltagged_unitpairs(self, file=None):
        """
            save in a file all pairs tagged with the discourse relation
            that related them
            unit1, unit2, rel, seq

        """
        all_rels = []
        for docidx in self.corpus.index:
            doc_hdrs = self.corpus.loc[docidx, "hdrs"]
            doc_edus = self.corpus.loc[docidx, "edustxt"]
            # doc_edus_idx = self.corpus.loc[docidx, "edus"]
            doc_rels = get_rels_in_doc(doc_hdrs, doc_edus)
            all_rels += doc_rels
        if file is not None:
            with open(file, "w") as outfile:
                outfile.writelines([";".join(x)+"\n" for x in all_rels])
        return all_rels

    def get_docs():
        """
            return documents (only txt)
        """
        return [d[1] for d in self.documents]

    def cluster(self, *params, algo="kmeans"):
        """
            Apply clustering to corpus and save clusters
        """
        # algo = "kmeans"
        if algo == "kmeans":
            clustering = kmeans([d[1] for d in self.documents],
                                self.corpus, params[0])
            self.clusters = clustering.clusters
            self.model = clustering.model
            self.cluster_inertia = clustering.inertia
            self.data = clustering.data
        # doc2vec
        if algo == "doc2vec":
            model = Doc2Vec.load("d2v.model")  # path 2 model
            print("Not impl. yet -> todo : integrate what i did in notebook")

        return clustering

    def serialize(self):
        """
            return list of data to be rendered in HTML
            0 : docid
            1 : text
            2 : units
            3 : ccidx
        """
        outputdata = []
        for (docid, text) in self.documents:
            text = self.corpus.loc[docid, "text"]
            units = self.corpus.loc[docid, "edustxt"]
            graph = self.corpus.loc[docid, "graph"]
            ccidx = get_root(graph)    # central unit idx
            outputdata.append([docid, text, units, ccidx])
        return outputdata

    def filter(self, docids):
        newdocuments = []
        for docid in docids:
            text = self.corpus.loc[docid, "text"]
            newdocuments.append((docid, text))
        self.documents = newdocuments


def load_dtf(files):
    # load filepath, docannot, docname, text and units into Dtf
    SciDTB = pd.DataFrame(columns=["filepath"], data=files)
    SciDTB["docname"] = SciDTB["filepath"].apply(get_docname)
    SciDTB["annot"] = SciDTB["filepath"].apply(get_json)
    SciDTB["segments"] = SciDTB["annot"].apply(get_edustxt)
    # SciDTB["text"] = SciDTB["segments"].apply(rebuild_text)
    # set index to docname
    SciDTB = SciDTB.set_index("docname")
    # get list of hdr
    SciDTB["hdrs"] = SciDTB["annot"].apply(get_hdrs)
    # get list of edus
    SciDTB["edus"] = SciDTB["annot"].apply(get_edus)
    # get list of tokens
    SciDTB["toks"] = SciDTB["annot"].apply(get_words)
    # edus text list
    SciDTB["edustxt"] = SciDTB["annot"].apply(get_edustxt)
    # get rels list
    SciDTB["rels"] = SciDTB["annot"].apply(get_rels)
    # concatnate text so that i can apply duplicate function
    # SciDTB["cleaned_txt"] = SciDTB["edustxt"].apply(rebuild_txt_clean)
    SciDTB["txt"] = SciDTB["edustxt"].apply(rebuild_txt)
    # get nb EDU in from annot
    SciDTB["nbedus"] = SciDTB["edus"].apply(lambda x: len(x))
    # get nb words from text
    SciDTB["nbtoks"] = SciDTB["toks"].apply(lambda x: len(x))
    # get nb of hdr
    SciDTB["nbrels"] = SciDTB["rels"].apply(lambda x: len(x))
    # get sentences
    SciDTB["sentences"] = SciDTB["txt"].apply(lambda x: [s for s in x.split("<S>") if s is not ""])
    # get sent ids for edus
    SciDTB["edus_sent_ids"] = SciDTB["edustxt"].apply(get_sent_idx)
    # get graph
    SciDTB["graph"] = SciDTB["hdrs"].apply(get_graph)
    return SciDTB
