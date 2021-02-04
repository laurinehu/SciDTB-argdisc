from itertools import chain, combinations
import networkx as nx
import networkx.algorithms.isomorphism as iso
import pandas as pd
from preprocessing_utils import draw_graph
from tqdm import tqdm


def get_nodes_combinations(graph):
    """
    returns all nodes combinations from a set of nodes
    """
    nodes = graph.nodes()
    nodes_powerset = []
    for n in chain.from_iterable(combinations(nodes, r) for r in range(len(nodes)+1)):
        if len(n) > 1:
            nodes_powerset.append(list(n))
    return nodes_powerset


def get_subgraphs(graph):
    """
        extract subgraphs from a given annot
    """
    nodes_powerset = get_nodes_combinations(graph)
    #print("Doing")
    #draw_graph(graph)
    subgraphs = []
    for nodes in nodes_powerset:
        subg = graph.subgraph(nodes)
        nodes = subg.nodes(data=True)
        if nx.is_weakly_connected(subg):
            subgraphs.append(subg)
    return subgraphs


def already_exists(sub, subs):
    """
    returns true if a schema is contained in a list of schema
    """
    exists = False
    em = iso.categorical_edge_match('label', 1)
    #nm = iso.categorical_node_match('type', "edu")
    for compsub in subs:
        if nx.is_isomorphic(compsub, sub, edge_match=em):
            exists = True
            break
    return exists


def find_schema_id(sub, dico):
    """
        find a schema id for a given schema
    """
    for key in dico.keys():
        em = iso.categorical_edge_match('label', 1)
        if nx.is_isomorphic(dico[key], sub, edge_match=em):
            return key


def to_doc_dic(doc_subs, schema_dic={}):
    """
    doc_subs is a list of list of subs
    """
    doc_attribs = {}
    schema_dic = {}
    id_attr = 0
    # loop over doc subs
    for docidx, subs in tqdm(enumerate(doc_subs)):
        doc_attribs[docidx] = []
        # for all subs of cur doc
        for sub in subs:
            # if schema not exist, create from attr idx
            if not already_exists(sub, schema_dic.values()):
                schema_dic[id_attr] = sub
                id_schema = id_attr
                id_attr += 1
            else:
                id_schema = find_schema_id(sub, schema_dic)
            doc_attribs[docidx].append(id_schema)
    return (doc_attribs.values(), schema_dic)


def to_ctx(doc_attribs, all_attribs):
    """
        create context of pairs :
    """
    dic_ctx = {}
    # loop on docs
    for doc_idx, attributes in enumerate(doc_attribs):
        dic_ctx[doc_idx] = []
        # attributes = doc_attrs[obj] # recover list of attr
        for attr in all_attribs:
            if attr in attributes:
                dic_ctx[doc_idx].append(1)
            else:
                dic_ctx[doc_idx].append(0)

    dtf = pd.DataFrame.from_dict(dic_ctx, orient="index", columns=all_attribs)
    return dtf
