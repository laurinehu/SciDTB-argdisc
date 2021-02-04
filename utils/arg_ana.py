import string
import networkx as nx


def get_root(graph):
    """
    recup noeud qui a un out dgre == 0
    """
    nodes = [n for n in graph]
    out_degs = [graph.out_degree(n) for n in graph]
    node_outdeg_dic = dict(zip(nodes, out_degs))
    # in_degs = [graph.in_degree(n) for n in graph]
    # loop nodes + out deg and rturn when its value is 0
    for k, v in node_outdeg_dic.items():
        if v == 0:
            return k

def get_arg(argsmap, eduidx):
    """
    return arg index of given edu
    calculated from argsmap
    """
    for adu, edus in argsmap.items():
        for edu in edus:
            if edu == eduidx:
                return adu

def get_arg_macrog(argsmap, graph):
    """
        returns discourse graph that connects arguments
    """
    outg = nx.DiGraph()
    for src, trg, data in graph.edges(data=True):
        argsrc = get_arg(argsmap, src)
        argtrg = get_arg(argsmap, trg)
        if argsrc != argtrg:
            if argsrc not in outg.nodes:
                outg.add_node(argsrc)
            if argtrg not in outg.nodes:
                outg.add_node(argtrg)
            outg.add_edge(argsrc, argtrg, label=data["label"])
    return outg


def get_arg_graph(argannot):
    """
    builds nx graph from arg annotation
    """
    G = nx.DiGraph()
    G.add_nodes_from(argannot.keys())
    for nodeid, nodedesc in argannot.items():
        if int(nodedesc["dirto"]) != 0:
            e = (nodeid, nodeid+int(nodedesc["dirto"]))
            lab = nodedesc["rel"]
            G.add_edge(*e, label=lab)
    # relabel nodes to start at 0
    nodes = G.nodes()
    mapping = {old:new for old, new in zip(nodes, map(lambda x:x-1, nodes))}
    out = nx.relabel_nodes(G, mapping)
    return out

def get_arg_subgraph(edus, graph):
    return graph.subgraph(edus)

def get_arg_subgraphs(adus_edus_mapping, graph):
    out = []
    for adu, edus in adus_edus_mapping.items():
        if adu is not None:
            out.append(graph.subgraph(edus))
    return out

def get_arg_subs(adusmapping, graph):
    out = []
    for aidx, eids in adusmapping.items():
        out.append(graph.subgraph(eids))
    return out

def get_json(G):
    out_dic = {"nodes":[],
               "links":[]}
    for n in G.nodes():
        curnode = {"id": n,
                   "name": n}
        out_dic["nodes"].append(curnode)
    for s, t, d in G.edges(data=True):
        label = d["label"]
        curedge = {"source":s,
                   "target":t,
                   "label": label}
        out_dic["links"].append(curedge)
    print(out_dic)
    return out_dic


def save(G, fname):
    out_dic = get_json(G)
    #print(out_dic)
    json.dump(out_dic,
              open(fname, 'w'), indent=2)
    print("Saved")


def parse_annot(iobtags):
    """
        from file in iobfileformat,
        create dict {argdescr:arg}
        argdescr = type-rel-dirto
    """
    args = {}
    sent = ""
    curtype = None
    i=0
    for line in iobtags:
        #print(">>>"+line)
        word, descr = line.split("\t")

        if word != "###":
            #print(descr)
            descr = descr.replace("\n", "")

            # to deal with "-" sep and "-" minus
            descr = descr.replace("--", "_~")
            descr = descr.replace("-", "_")
            descr = descr.replace("~", "-")

            word = word+" "

            iobt, atype, arel, aattach = descr.split("_")

            if aattach == 0:
                print("YES")
                print(sent)

        if iobt == "B" or word == "###":
            # init case
            if sent == "":
                curattach = aattach
                curtype = atype
                currel = arel
                sent = word

            # add if not init case
            else:
                args[i] = {"sent":sent,
                           "type": curtype,
                           "rel" : currel,
                           "dirto": curattach}
                sent = word
                curattach = aattach
                curtype = atype
                currel = arel
            i+=1
        else:
            sent = sent+ word
    return args


def parse_iob(iobtags):
    """
        from file in iobfileformat,
        create dict {argdescr:arg}
        argdescr = type-rel-dirto
    """
    args = {}
    sent = ""
    curtype = None
    for line in iobtags:
        #print(">>>"+line)
        word, descr = line.split("\t")

        if word != "###":
            #print(descr)
            descr = descr.replace("\n", "")

            # to deal with "-" sep and "-" minus
            descr = descr.replace("--", "_~")
            descr = descr.replace("-", "_")
            descr = descr.replace("~", "-")

            word = word+" "

            iobt, atype, arel, aattach = descr.split("_")

        if iobt == "B" or word == "###":
            # init case
            if sent == "":
                curtype = atype
                sent = word

            # add if not init case
            else:
                # add cursent
                if curtype not in args.keys():
                    args[curtype] = [sent]
                else:
                    args[curtype].append(sent)
                sent = word
                curtype = atype
        else:
            sent = sent+ word
    return args


def get_arg_annot(filename):
    with open(filename) as iobfile:
        iobtags = iobfile.readlines()
        args = parse_annot(iobtags)
        # atype = parse_iob(iobtags)
    # return atype
    return args


def get_arg_sents(arg_annot):
    """
    recover list of arguments and type
    """
    sents = []
    for k, arg in arg_annot.items():
        sents.append(arg["sent"])
    return sents


def get_arg_types(arg_annot):
    """
    recover list of arguments and type
    """
    types = []
    for k, arg in arg_annot.items():
        types.append(arg["type"])
    return types


def remove_nones(dic):
    out = {}
    for k, v in dic.items():
        if k is not None:
            out[k] = v
    return out
## Edus / Adus mapping

def edus_per_adus(adus, edus):
    """
        return EDUs segments as text, classified based on arguments
        list of list : args and edus of arg
    """
    outlst = [[]]
    for adu in adus:
        for edu in edus:
            # test edu in adu
            test = True
            for w in edu:
                if w not in adu:
                    test = False
            # if edu in adu, append
            if test == True:
                outlst[-1].append(edu)
        outlst.append([])
    return outlst

def to_sent_edus_mapping(lst_sent_idx):
    mapping = {}
    for idx, sentidx in enumerate(lst_sent_idx):
        if sentidx not in mapping.keys() and sentidx is not None:
            mapping[sentidx] = [idx]
        elif sentidx is not None:
            mapping[sentidx].append(idx)
    return mapping

def to_edus_adus_mapping(adus, edus):
    """
        recover edus/adus mapping
        dict : key, val where key is an edu idx,
                              val is an adu idx
    """
    out = {}
    args_done = []
    aidx = 0
    aduscp = adus.copy()
    while len(aduscp) != 0:
        #print(aduscp)
        atxt = aduscp.pop(0)
        #print(atxt)
    #for aidx, atxt in enumerate(adus):
        seen = []
        for eidx, etxt in enumerate(edus):
            #print(etxt)
            if etxt in atxt:
                if eidx not in out.keys() and etxt not in seen:
                    out[eidx] = aidx
                    seen.append(etxt)

        aidx+=1

    # check all edus have a corresponding arg ?
    for eidx, edu in enumerate(edus):
        if eidx not in out.keys():
            out[eidx] = None

    return out

def edus_adus_mapping(adus, edus):
    """
        recover edus/adus mapping
        dict : key, val where key is an edu idx,
                              val is an adu idx
    """
    out = {}
    for aidx, adu in enumerate(adus):
        for eidx, edu in enumerate(edus):
            # test edu in adu
            test = True
            for w in edu:
                if w not in adu:
                    test = False
            # if edu in adu, append
            if test == True:
                out[eidx] = aidx

    # check all edus have a corresponding arg ?
    for eidx, edu in enumerate(edus):
        if eidx not in out.keys():
            out[eidx] = None

    return out


def nb_edus_per_adu(edus_adus_mapping):
    count_edus = {k:0 for k in edus_adus_mapping.values()}
    for edu, adu in edus_adus_mapping.items():
        count_edus[adu]+=1
    return count_edus

## Write args to file
# Arg : Type

def write_args(args, types, outfile):
    lines = []
    for a, t in zip(args, types):
        lines.append(a+"\t"+t+"\n")
    with open(outfile, "w") as out:
        out.writelines(lines)


def to_adus_edus_mapping(edus_adus_mapping):
    """
        returns a dict : adu:[edulist]
        from edus mapping
    """
    outdic = {}
    for edu, adu in edus_adus_mapping.items():
        if adu not in outdic.keys():
            outdic[adu] = [edu]
        else:
            outdic[adu].append(edu)
    return outdic


def recover_adu(edu, mapping):
    # recover adu
    aduidx = mapping[edu]
    return aduidx


def recover_adu_edus(adu, mapping):
    """
        from an adu idx, recover set of corresponding edus
    """
    out_edus = []
    for cedu, cadu in mapping.items():
        if cadu == adu:
            out_edus.append(cedu)
    return out_edus

def has_sim_seg(adus, sents):
    out = True # out is a dict of argument and yes or no cond

    for ak, av in adus.items():
        for sk, sv in sents.items():# and sents[k] == v:
            if av == sv:# sublist(av, sv):# == sv:
                out = True
                break
            else:
                out = False
    return out

def sim_seg(adus, sents):
    out = {} # out is a dict of argument and yes or no cond

    for ak, av in adus.items():
        for sk, sv in sents.items():# and sents[k] == v:
            if av == sv:#sublist(av, sv):# == sv:
                out[ak] = True
                break
            else:
                out[ak] = False
    return out

def prop_sim_seg(simsegs):
    out = 0 # out is a dict of argument and yes or no cond
    for k, v in simsegs.items():
        if v == True:
            out+=1
    return out/len(simsegs)

def sublist(lst1, lst2):
    return set(lst1) <= set(lst2)


def get_adu_mapping(adus, sents):
    adus_map = {}
    for aidx, adu in adus.items():
        for sidx, sent in sents.items():
            if sublist(sent, adu):
                if aidx not in adus_map.keys():
                    adus_map[aidx] = [sidx]
                else:
                    adus_map[aidx].append(sidx)
    return adus_map

def get_sent_mapping(adus, sents):
    sents_map = {}
    for sidx, sent in sents.items():
        for aidx, adu in adus.items():
            if sublist(adu, sent):
                if sidx not in sents_map.keys():
                    sents_map[sidx] = [aidx]
                else:
                    sents_map[sidx].append(aidx)
    return sents_map


def flat(l):
    return [item for sublist in l for item in sublist]

def merge_dict(dicts):
    out = {}
    for dic in dicts:
        for k, v in dic.items():
            if k not in out:
                out[k] = [v]
            else:
                out[k].append(v)
    return out

def find_edge_label(ex, ey, edges):
    for x, y,data in edges:
        if x == ex and y == ey:
            return data["label"]
    return None

def rels_mapping(mg, ag):
    aedges = ag.edges(data=True)
    aedges_rels = [(x,y) for x,y,lab in aedges]
    aedges_labels = [x[2]["label"] for x in aedges]
    medges = mg.edges(data=True)
    mapping = {}

    for idx, edge in enumerate(medges):
        #print(edge)
        ex, ey, lab = edge[0], edge[1], edge[2]["label"]
        if (ex,ey) in aedges_rels and lab is not None:
            # get index of ex ey in medges
            label = find_edge_label(ex, ey, aedges)
            if lab not in mapping.keys():
                mapping[lab] = [label]
            else:
                mapping[lab].append(label)
    return mapping


def elab_root_rels(arg, disc):
    root = get_root(arg)
    out = []
    for a, b, data in disc.edges(data=True):
        if data["label"] == "elab-addition" and b == root:
            try:
                arg_rel =  arg.get_edge_data(a, b)["label"]
            except:
                arg_rel = None
            out.append(arg_rel)
    return out
