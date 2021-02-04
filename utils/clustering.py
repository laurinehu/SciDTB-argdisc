import glob
import pandas as pd
import importlib
import re
from utils import export_to_json
from corpus_utils import Corpus, load_dtf

# Import corpus to dataframe (text and units)

# the corpus is organized following 3 directories
# in each directory, some files are annotated by multiple annotators
# so we take all the uniq documents (just 1 annotator)
pathtodirtrain = "../data/SciDTB/train/"
pathtodirtest = "../data/SciDTB/test/gold/"
pathtodirdev = "../data/SciDTB/dev/gold/"
pathtographs = "../data/SciDTB/graphimages/"
path2properties = "../data/SciDTB_articles_properties.csv"

s
# considere only 1 annotator (taken randomly)
# /!\ sera à regarder proprement selon les exéps à venir
allano = "*.dep"
files = glob.glob(pathtodirtrain+allano)  # for train keep only one annotator
# remove duplicate documents (other annotators)
files = [x for x in files if "anno2" not in x and "anno3" not in x]
files += glob.glob(pathtodirtest+allano)
files += glob.glob(pathtodirdev+allano)

SciDTB = load_dtf(files)

# ids =  list(SciDTB.index.values)
# texts = [ SciDTB.loc[idd,"text"] for idd in ids]
c = Corpus(SciDTB)


nbclusters = 20  # size of clustering
clustering = c.cluster(nbclusters, algo="kmeans")
clustering.clusters.sort(key=lambda x: len(x.texts))

clusttokeep = 0
sentences = []
for (text, docid) in clustering.clusters[clusttokeep].texts:
    edus = SciDTB.loc[docid, "edustxt"]
    cursent = ""
    for edu in edus:
        edu = edu.replace("\r", "")
        cursent += edu
        # case end of senttence
        if edu.endswith("<S>"):
            sentences.append(cursent)
            cursent = ""
print(sentences)

with open("clustevent_sents.txt", "w") as outputfile:
    outputfile.writelines([x+"\n" for x in sentences])
