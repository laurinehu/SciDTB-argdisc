import sys
import math
import networkx as nx
import networkx.algorithms.isomorphism as iso

sys.path.append('utils/gSpan/')

from gspan_mining.gspan import gSpan


def gspan(graphs, minsup, minvert, method=1):
    """
    """
    gs = gSpan(
            nxg_db=graphs,
            min_support=minsup,
            min_num_vertices=minvert,
            max_num_vertices=math.inf,
            max_ngraphs=math.inf,
            #visualize=True,
            where=True,
            method=method)
    gs.run()

    return (gs.graphs, gs._report_df)


def get_sub_types(sub, doc_graphs, doc_types):
    """
    from a subgraph, a list of arg graphs, and a list of arg types,
    recovers the count of args that are isomorph for each arg type
    """
    out = {}
    #for graphs, types in zip(doc_graphs, doc_types):
    for g, t in zip(doc_graphs, doc_types):
        if nx.is_isomorphic(sub, g):
            if t in out.keys():
                out[t]+=1
            else:
                out[t] = 0
    return out


def gspangraph_to_nxgraph(gspangraph):
    outg = nx.Graph()
    # recover and append vertices to nxgraph
    vertices = list(gspangraph.vertices.keys())
    outg.add_nodes_from(vertices)
    # recover edges and append to nxgraph
    edges = dict(gspangraph.set_of_elb)
    for lab, edges in edges.items():
        edges = list([frozenset(x) for x in list(edges)])
        edges = set(edges)
        for edge in edges:
            edge = list(edge)
            outg.add_edge(edge[0], edge[1], label=lab)
    return outg
