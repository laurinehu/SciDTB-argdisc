from sentence_transformers import SentenceTransformer
from itertools import combinations
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances


def save_embeddings(inputfile, outputfile):
    with open(inputfile, "r") as sentsfile:
        sents = sentsfile.readlines()
        sents = [x.replace("\n", "").lower() for x in sents]
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embedded_sents = model.encode(sents)
    for sent in embedded_sents:
        print(type(sent))
        print(str(len(sent)))
        print(sent)


def savedistances(intputfile, outputfile):
    """
        load units from file
        calculate distances
        sort from nearest to farthest
        save to file
    """

    # load units and embedd
    with open(intputfile, "r") as unitsfile:
        units = unitsfile.readlines()
        units = [x.lower() for x in units]

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embedded_sent = model.encode(units)

    # calculate distances (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances)
    # pairwise_dists = pairwise_distances(embedded_sent)
    cosine_pairwise_dists = cosine_distances(embedded_sent)

    dists = cosine_pairwise_dists

    # write unit similarity to file
    lines = []
    # matrix rows
    for i in range(len(dists)):
        # matrix cols
        for j in range(len(dists[i])):
            iunit = units[i]
            iunit = iunit.replace("\n", "")
            junit = units[j]
            junit = junit.replace("\n", "")
            dist = dists[i][j]
            lines.append(str(dist)+"\t"+iunit + "\t" + junit + "\n")

    with open(outputfile, "w") as output:
        output.writelines(lines)


def keep_sim(unitsdistfile, outputfile):
    """
        takes as input a file with distance \t u1 \t u2 on each line
        keep only unit pairs that have a cosine distance < 0
    """
    with open(unitsdistfile, "r") as unitsdistfile:
        unitpairs = unitsdistfile.readlines()
        unitpairs = [x.replace("\n", "") for x in unitpairs]

    print("Checking unit pairs")
    print("dbg")
    print(unitpairs[0].split("\t"))
    newfilelines = []
    for unitpair in unitpairs:
        (dist, u1, u2) = unitpair.split("\t")
        # check if some similarity
        if float(dist) < 0:
            print("Yeyyyy, similar units "+dist)
            newfilelines.append(dist+"\t"+u1+"\t"+u2+"\n")

    with open(outputfile, "w") as outputfile:
        outputfile.writelines(newfilelines)


def debug(sentences):

    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embedded_sent = model.encode(sentences)

    # calculate distances (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances)
    # pairwise_dists = pairwise_distances(embedded_sent)
    cosine_pairwise_dists = cosine_distances(embedded_sent)

    dists = cosine_pairwise_dists

    # write unit similarity to file
    lines = []
    # matrix rows
    for i in range(len(dists)):
        # matrix cols
        for j in range(len(dists[i])):
            iunit = sentences[i]
            iunit = iunit.replace("\n", "")
            junit = sentences[j]
            junit = junit.replace("\n", "")
            dist = dists[i][j]
            print(str(dist)+"\t"+iunit + "\t" + junit)


if __name__ == '__main__':
    # unitsfile = "cluster_20_0.txt"
    # unitsfile = "all_units.txt"
    # outputfile = "all_units_cosinedist.txt"
    """
    sentences = ["This framework generates embeddings for each input sentence",
                 "This tool generates embeddings for each input text segment",
                 "This work generates vectors for each segment of the list",
                 "I had a pizza",
                 "Vectors are good",
                 "Vectors are bad",
                 "My diner was bad"
                 ]
    """
    #debug(sentences)

    infile = "16classes_10datapoints_discrelsent.txt"
    outfile = "discrrelsent_distances.txt"
    # keep_sim(infile, outfile)
    save_embeddings(infile, outfile)
