import pandas as pd
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from preprocessing_utils import (get_annot, get_docname, get_json,
                                get_hdrs, get_edus, get_words,
                                get_edustxt, get_rels, get_graph,
                                draw_graph, rebuild_txt, get_coarse_rels,
                                get_root_nodes, get_root, get_rels_per_level,
                                split_annot)

def split_moves(graph, edustxt):
    """
    code dupliqué, a merger proprement
    """
    out = {"bg": [],
           "method": [],
           "eval": []}

    # all rels =/= eval or bg are part of method
    method_rels = ['progression', 'comparison', 'enablement',
                   'elab', 'manner', 'summary', 'attribution',
                   'result', 'contrast', 'condition', 'exp',
                   'same', 'null', 'joint', 'temporal', 'cause']


    # recover root node and add to "elab" part
    root_node = get_root(graph)
    out["method"].append(root_node)

    # get root neighbors
    root_out_edges = list([x for x in graph.in_edges(root_node, data=True)])

    # pour chaque arc dirigé vers la racine
    for out_edge in root_out_edges:
        src = out_edge[0]
        trg = out_edge[1]
        cur_rel = out_edge[2]["label"]  # recov rel label
        # convert 2 coarse grain if fine grain
        cur_rel = cur_rel.split("-")[0] if "-" in cur_rel else cur_rel
        # recover sons of cur out edge
        sons = [x for x in nx.bfs_edges(graph, reverse=True, source=src)]
        sons = list(set([i for sub in sons for i in sub]))
        if sons == []: sons.append(src)

        # add method sons
        if cur_rel in method_rels: out["method"] += sons
        if cur_rel == "evaluation": out["eval"] += sons
        if cur_rel == "bg": out["bg"] += sons

    return out

def get_sub(g, edus):
    """
        returns networkx subgraph
    """
    return g.subgraph(edus)


def get_text(edus, movesedus):
    """
        rebuilds text from edus
    """
    txt = ""
    for i in movesedus:
        txt += edus[i]
    return txt.replace("<S>", "")


def get_sents_txt(edus):
    """
    recover index of sentence for each edu
    """
    sentences = []
    cursent = ""
    for edu in edus:
        edu = edu.replace("\r", "")
        cursent+=edu
        if edu.endswith(". <S>"):
            sentences.append(cursent)
            cursent = ""
    return sentences


def sents_txt(sidxs, edusidx, segments):
    prec = 0
    sent = ""
    sents = []
    #last = len(sidxs)

    for idxedu, idxsent in enumerate(sidxs):
        try:
            seg = segments[edusidx[idxedu]]
        except IndexError:
            seg = ""
        if idxsent == prec:
            #print(segments[edusidx[idxedu]])
            sent+=seg

        # nouvelle phrase
        elif idxsent != prec:
            prec = idxsent
            sents.append(sent)
            sent = seg

        # nouvelle phrase ou dernier element
        if idxedu == len(sidxs)-1:
            sents.append(sent)
    return sents

def sents_graphs(sidxs, edusidx, segments, graph):
    out_graphs = []
    curg = []
    prec = 0

    for idxedu, idxsent in enumerate(sidxs):
        try:
            idx = edusidx[idxedu]
        except IndexError:
            idx = None

        if idx is not None:
            if idxsent == prec:
                curg.append(edusidx[idxedu])
            # nouvelle phrase
            elif idxsent != prec:
                prec = idxsent
                out_graphs.append(graph.subgraph(curg))
                curg=[edusidx[idxedu]]

            # nouvelle phrase ou dernier element
            if idxedu == len(sidxs)-1:
                out_graphs.append(graph.subgraph(curg))

    return out_graphs


def sidx(moves, movesidx, sentidx):
    if movesidx != []:
        movesidx.sort()
        mini = movesidx[0]
        maxi = movesidx[-1]+1
        out = sentidx[mini:maxi]  # return
        if out != []:
            # recalc to 0
            rem = 0+out[0]
            out = [x-rem for x in out]
        else:
            out=[]
    else:
        out = []
    return out

def dbg(movesdtf, x):
    print("segments")
    for s in movesdtf.loc[x, "segments"]:
        print(s)

    print("indexes")
    print("\t__bg__")
    print(movesdtf.loc[x, "sidxbg"])
    print("\t__method__")
    print(movesdtf.loc[x, "sidxmethod"])
    print("\t__eval__")
    print(movesdtf.loc[x, "sidxeval"])

    print("\n graphs")
    print("\t__bg__")
    draw_graph(movesdtf.loc[x, "gbg"])
    #print(movesdtf.loc[x, "sbg"])
    print("\t__method__")
    draw_graph(movesdtf.loc[x, "gmethod"])
    #print(movesdtf.loc[x, "sme"])
    print("\t__eval__")
    draw_graph(movesdtf.loc[x, "geval"])
    #print(movesdtf.loc[x, "seval"])
    print("__graphs__")
    print("method")
    for g in movesdtf.loc[x, "sgme"]:
        draw_graph(g)
    print("bg")
    for g in movesdtf.loc[x, "sgbg"]:
        draw_graph(g)
    print("eval")
    for g in movesdtf.loc[x, "sgeval"]:
        draw_graph(g)

def moves_dtf(corpus):
    """
        from a corpus dtf, create a moves dtf
        with moves txt, moves subgraphs, moves edus
    """

    columns = ["bg", "method", "eval", "gbg", "gmethod", "geval"]

    newdtf = pd.DataFrame(columns=columns)
    newdtf["moves"] = corpus.apply(lambda x : split_moves(x["graph"], x["edustxt"]), axis=1)

    newdtf["edus_sent_ids"] = corpus["edus_sent_ids"]  # recover edus
    newdtf["graph"] = corpus["graph"]
    newdtf["segments"] = corpus["segments"].apply(lambda x : [y.replace("\r", "").replace("<S>", "") for y in x])

    newdtf["bg"] = newdtf["moves"].apply(lambda x : [y for y in x["bg"]])
    newdtf["method"] = newdtf["moves"].apply(lambda x : [y for y in x["method"]])
    newdtf["eval"] = newdtf["moves"].apply(lambda x : [y for y in x["eval"]])

    # recover list of units for each move
    newdtf["ubg"] = newdtf.apply(lambda row: row.moves["bg"], axis=1)
    newdtf["ume"] = newdtf.apply(lambda row: row.moves["method"], axis=1)
    newdtf["ueval"] = newdtf.apply(lambda row: row.moves["eval"], axis=1)


    # recover subgraph for each move
    newdtf["gbg"] = newdtf.apply(lambda row : get_sub(row.graph, row.bg), axis=1) # get_sub(newdtf["graph"], newdtf["bg"])  #get_sub(dtf["graph"], dtf["bg"])
    newdtf["gmethod"] = newdtf.apply(lambda row : get_sub(row.graph, row.method), axis=1) # get_sub(newdtf["graph"], newdtf["bg"])  #get_sub(dtf["graph"], dtf["bg"])
    newdtf["geval"] = newdtf.apply(lambda row : get_sub(row.graph, row.eval), axis=1) # get_sub(newdtf["graph"], newdtf["bg"])  #get_sub(dtf["graph"], dtf["bg"])

    # recover text for each move
    newdtf["tbg"] = newdtf.apply(lambda row: get_text(row.segments, row.bg), axis=1)
    newdtf["tmethod"] = newdtf.apply(lambda row: get_text(row.segments, row.method), axis=1)
    newdtf["teval"] = newdtf.apply(lambda row: get_text(row.segments, row.eval), axis=1)

    # recover sentences for each move
    newdtf["sidxbg"] = newdtf.apply(lambda row: sidx(row, row.moves["bg"], row.edus_sent_ids), axis=1)
    newdtf["sidxmethod"] = newdtf.apply(lambda row: sidx(row, row.moves["method"], row.edus_sent_ids), axis=1)
    newdtf["sidxeval"] = newdtf.apply(lambda row: sidx(row, row.moves["eval"], row.edus_sent_ids), axis=1)

    # recover list of sentences
    newdtf["sbg"] = newdtf.apply(lambda row: sents_txt(row.sidxbg, row.moves["bg"], row.segments), axis=1)
    newdtf["sme"] = newdtf.apply(lambda row: sents_txt(row.sidxmethod, row.moves["method"], row.segments), axis=1)
    newdtf["seval"] = newdtf.apply(lambda row: sents_txt(row.sidxeval, row.moves["eval"], row.segments), axis=1)

    # recover sentence graphs
    newdtf["sgbg"] = newdtf.apply(lambda row: sents_graphs(row.sidxbg, row.moves["bg"], row.segments, row.graph), axis=1)
    newdtf["sgme"] = newdtf.apply(lambda row: sents_graphs(row.sidxmethod, row.moves["method"], row.segments, row.graph), axis=1)
    newdtf["sgeval"] = newdtf.apply(lambda row: sents_graphs(row.sidxeval, row.moves["eval"], row.segments, row.graph), axis=1)

    return newdtf


def rels_count(sub):
    """
        extract nb distinct relations
    """
    #nx.draw_networkx_edge_labels(sub)
    edges = sub.edges(data=True)
    rels = []
    for e in edges:
        rels.append(e[2]["label"])
    return Counter(rels)


def uniq_rels(doc_rels):
    """
        returns list of uniq rels from dic of rel count per move
    """
    outrels = []
    for move in doc_rels:
        for k, v in move.items():
            if k not in outrels:
                outrels.append(k)
    return outrels


def glob_uniq_rels(corpus_uniqrels):
    outrels = []
    for doc, rels in corpus_uniqrels.items():
        for rel in rels:
            if rel not in outrels:
                outrels.append(rel)
    return outrels


def get_moves_sent_rels(doc, rels):
    """
        extract rels in sentences for all moves
    """
    glob = {"method": get_sent_rels(doc, rels, move="me"),
            "bg": get_sent_rels(doc, rels, move="bg"),
            "eval": get_sent_rels(doc, rels, move="eval")
           }
    return glob


def get_sent_rels(doc, uniqrels, move="me"):
    """
        extract rels in sentences for a given move
    """
    rels_count = {k: 0 for k in uniqrels}
    docsentgraph = doc["sg"+move]  # recover sentence graphs for move
    for g in docsentgraph:
        sentedges = g.edges.data()
        sentedgelist = [z["label"] for (x, y, z) in sentedges]
        for edges in sentedgelist:
            rels_count[edges] += 1
    return rels_count


def sents_rels_dtf(movesdtf, uniqrels, docs):
    out = {k:[] for k in docs}
    names = [d+r for d in ("m_", "b_", "e_") for r in uniqrels]
    for row in movesdtf.iterrows():
        data = get_moves_sent_rels(row[1], uniqrels)
        for move, count in data.items():
            for rel in uniqrels:
                if rel in count.keys():
                    out[row[0]].append(count[rel])
                else:
                    out[row[0]].append(count[rel])
    return pd.DataFrame.from_dict(out, orient='index', columns=names)


def rels_dtf(movesdtf):
    out = {}
    uniq = {}

    # recover dic of rels for each move
    # recover list uniq rels for each doc
    for row in movesdtf.iterrows():
        data = row[1]
        (gme, gbg, gev) = data["gmethod"], data["gbg"], data["geval"]
        out[row[0]] = [rels_count(gme), rels_count(gbg), rels_count(gev)]
        uniq[row[0]] = uniq_rels(out[row[0]])


    # create name list for new dtf
    globuniq = glob_uniq_rels(uniq)
    names = [d+r for d in ("m_", "b_", "e_") for r in globuniq]
    out_dic = {}

    # create output dic
    for row in movesdtf.iterrows():
        data = row[1]
        out_dic[row[0]] = []
        (gme, gbg, gev) = data["gmethod"], data["gbg"], data["geval"]
        (cme, cbg, cev) = [rels_count(gme), rels_count(gbg), rels_count(gev)]
        # loop moves count
        for rel_count, t in [(cme, "m"), (cbg, "b"), (cev, "e")]:
            for rel in globuniq:
                if rel in rel_count.keys():
                    out_dic[row[0]].append(rel_count[rel])
                else:
                    out_dic[row[0]].append(0)

    return pd.DataFrame.from_dict(out_dic, orient='index', columns=names), globuniq


def plot_moves_rels_distrib(rels_dtf, uniqrels, norm=False):
    # loop over each move
    bars_m, bars_b, bars_e = [], [], []
    for group in ("m", "b", "e"):
        lst = []
        # for all rels
        for rel in uniqrels:
            # recover column header
            h = group+"_"+rel
            # add nb rel
            lst.append(sum(list(rels_dtf[h])))
        if group == "m": bars_m = lst
        elif group == "b": bars_b = lst
        elif group == "e": bars_e = lst

    bars = np.add(bars_m, bars_b).tolist()
    r = range(len(bars_m))
    names = uniqrels
    barWidth = 1

    if norm :
        totals = [i+j+k for i, j, k in zip(bars_m, bars_b, bars_e)]
        bars_m = [i/j*100 if j != 0 else 0 for i, j in zip(bars_m, totals)]
        bars_b = [i/j*100 if j != 0 else 0 for i, j in zip(bars_b, totals)]
        bars_e = [i/j*100 if j != 0 else 0 for i, j in zip(bars_e, totals)]


    # Create brown bars
    plt.bar(r, bars_m, color='#7f6d5f', edgecolor='white', width=barWidth)
    # Create green bars (middle), on top of the firs ones
    plt.bar(r, bars_b, bottom=bars_m, color='#557f2d', edgecolor='white', width=barWidth)
    # Create green bars (top)
    plt.bar(r, bars_e, bottom=np.add(bars_m, bars_b), color='#2d7f5e', edgecolor='white', width=barWidth)

    # Custom X axis
    plt.xticks(r, names, fontweight='bold', rotation=90)
    plt.xlabel("group")
    plt.legend(["method", "bg", "eval"])

    # Show graphic
    plt.show()
