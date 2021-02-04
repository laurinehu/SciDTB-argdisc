import json
from networkx.readwrite import json_graph


def save_doc_attrs(outfilename, attribs):
    with open(outfilename, "w") as out_attribs:
        for doc in attribs:
            attrs = [str(x) for x in doc]
            out_attribs.write(";".join(attrs)+"\n" if doc != [] else "\n")
            

def save_doc_names(outfilename, names):
    with open(outfilename, "w") as out_names:
        out_names.writelines([x+"\n" for x in names])
        

def save_g(G, fname):
    json.dump(json_graph.node_link_data(G),
              open(fname, 'w'), indent=2)

def load_g(fname):
    with open(fname) as f:
        jg = json.load(f)
    return json_graph.node_link_graph(jg)
    
def save_nx_subs(dic):
    # loop all subs and dump to file
    for k,v in dic.items():
        outfile = "sub_"+str(k)+".json"
        save_g(v, outfile)
        
def load_nx_subs(files):
    out_subs = {}
    for f in files:
        subidx = f.replace(".json", "")
        subidx = subidx.replace("sub_", "")
        subidx = int(subidx)
        out_subs[subidx] = load_g(f)
    return out_subs