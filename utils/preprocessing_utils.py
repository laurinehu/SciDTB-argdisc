import os, re
import json
import string
import pandas as pd
import networkx as nx
#import pygraphviz
from nltk import word_tokenize, sent_tokenize
from IPython.display import Image  # to print graphs directly in jupyter note
from IPython.display import display
# for crawler
import requests
#from bs4 import BeautifulSoup  # crawl infos

from nltk.corpus import stopwords

def lower_nopunc_tok(sent):
    # remove punkt
    sent = sent.replace("<S>", "")
    sent = sent.replace("-RRB- ", "")
    sent = sent.replace("-LRB- ", "")
    sent = sent.replace(" '", "")
    mypunc = string.punctuation+ '’"”“'
    sent = sent.translate(str.maketrans('', '', mypunc))
    # lower
    #sent = sent.lower()
    # tokenize
    sent = word_tokenize(sent)
    return sent

def preprocess(sents):
    """
         text is a list of sent/adus forming the document
    """
    newsents = []
    for sent in sents:
        newsent = " ".join(lower_nopunc_tok(sent))
        newsent = newsent.replace("RRB", "")
        newsent = newsent.replace("LRB", "")
        newsents.append(newsent)
    return newsents

def meta_type(gbg, gmethod, geval):
    """
    0 : B,E,M
    1 : B, M
    2 : E, M
    3 : M
    """
    if len(gbg) != 0 and len(gmethod) != 0 and len(geval) != 0:
        return 0
    elif len(gbg) != 0 and len(gmethod) != 0 and len(geval) == 0:
        return 1
    elif len(geval) != 0 and len(gmethod) != 0 and len(gbg) == 0:
        return 2
    elif len(gmethod) != 0 and len(gbg) == 0 and len(geval) == 0:
        return 3

################
# CORPUS INFO
################

def get_text(filepath):
    '''
    recover text from filename
    '''
    with open(filepath) as txt:
        return txt.read()


def get_docname(filepath):
    '''
    recover documentname
    '''
    name = os.path.basename(filepath)
    name = name.split(".")[0]
    return name


def get_json(filepath):
    '''
    recover file in json format
    '''
    with open(filepath, encoding='utf-8-sig') as file:
        # txt = file.read().decode('utf-8-sig')
        jsonannot = json.load(file)
        # jsonannot = json.loads(txt)
        return jsonannot


def get_hdrs(jsondoc):
    '''
    get hdr description for each doc
    '''
    hdr_lst = [(x["id"], x["parent"], x["relation"]) for x in jsondoc['root'] if x["id"] != 0]
    # hdr_lst = hdr_lst.pop() # remove first item witch rpz root
    return hdr_lst


def get_rels(jsondoc):
    '''
    get  list of rels from a document
    '''
    rels = [x['relation'] for x in jsondoc['root'] if x['relation'] != "null"]
    return rels


def get_edus(jsondoc):
    '''
    get list  of edus
    '''
    edus = [x['id'] for x in jsondoc['root'] if x["id"] != 0]
    return edus


def get_edustxt(jsondoc):
    '''
    get edus text
    '''
    edus = [x['text'] for x in jsondoc['root'] if x['text'] != "ROOT"]
    return edus


def get_words(jsondoc):
    '''
    get words
    '''
    edustxt = get_edustxt(jsondoc)  # recover all edus
    toks = []  # prepare list where to append tokens
    for edutxt in edustxt:
        edutxt = edutxt.lower()
        trantab = str.maketrans("", "", string.punctuation)  # define transtab
        txtnopunc = edutxt.translate(trantab)  # removes punctuation by trans
        toks += word_tokenize(txtnopunc)                        # tokenizes
    toks = [x for x in toks if x != "root"]  # remove root edu
    return toks


def get_nbwords(jsondoc):
    '''
    get nb words
    '''
    toks = get_words(jsondoc)
    return len(toks)


def get_sent_idx(edus):
    """
    for all edus, recover id of sentence in document
    """
    sentidx = []
    # for edus in edusperdoc:
    curidx = 0
    for edu in edus:
        edu = edu.replace("\r", "")
        sentidx.append(curidx)
        if edu.endswith("<S>"):
            curidx += 1
    return sentidx


def get_rels_in_doc(hdrs, edus_txt):
    """
    pour un document, récupère toutes les paires de phrases qui ont une relation
    prnd en argument la liste des hdrs du document
    """
    units_rels = []
    for (dep, head, rel) in hdrs:
        # 0 is head
        # 1 is dependent
        # 2 is rel tag
        if rel != "ROOT":  # do not consider root
            # rmove 1 for searchin index (bcs we do not consider root)
            conseq = 0
            seq = 0
            # test si unités consécutives
            if dep == head+1 or dep+1 == head:
                conseq = 1
            # test si unités séquentielles
            if dep > head:
                seq = 1
            depedu = edus_txt[dep-1].replace("\r", "")  # remove \r
            headedu = edus_txt[head-1].replace("\r", "")
            units_rels.append([depedu, headedu, rel, str(conseq), str(seq)])
    return units_rels


################################
# CRAWLER
################################

def get_html_content(docid):
    """
    crawl web to get authors
    """
    # regpat = "(.*-.*)_" # to capture id of document
    # idd = re.findall(regpat, docname)[0]
    #  print("https://www.aclweb.org/anthology/"+idd+"/")
    page = requests.get("https://www.aclweb.org/anthology/"+docid+"/")

    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


def get_properties(html_head):
    """
    from an html head from ACL anthology,
    extract info of interest and put it into a dict
    """
    # regex for info capturing
    # to apply on head of crawled html
    regproceedings = r'.*\"(.*?)\" name=\"citation_conference_title\"'
    regtitle = r'.*\"(.*?)\" name=\"citation_title\"'
    regauthors = r'content=\"([^\"]*)\" name=\"citation_author\"'
    regpdf = r'.*\"(.*?)\" name=\"citation_pdf_url\"' # recover link2pdf
    regpublidate = r".*\"(.*?)\" name=\"citation_publication_date\""

    title = re.search(regtitle, html_head).group(1)
    authors = re.findall(regauthors, html_head)
    publidate = re.search(regpublidate, html_head).group(1)
    proceedings = re.search(regproceedings, html_head).group(1)
    link2pdf = re.search(regpdf, html_head).group(1)

    out = {"title": title,
           "authors": authors,
           "publidate": publidate,
           "proceedings": proceedings,
           "link2pdf": link2pdf}

    return out


def crawl_info(SciDTB):
    """
        from a dataframe containing corpus, crawl descriptive informations on
        acl anthology website and merge it to existing dtf
    """
    corpus_properties = pd.DataFrame()  # dtf where to save properties
    for iddoc in SciDTB.index:
        idd = iddoc.split("_")
        idd = idd[0]
        print(idd)
        # regpat = "(.*-.*)_" # to capture id of document
        #  idd = re.findall(regpat, docname)[0]
        html_content = get_html_content(idd)
        data = get_properties(str(html_content.head))
        data["index"] = iddoc  # add futur index id
        # append to properties dict
        corpus_properties = corpus_properties.append(data, ignore_index=True)
    new = pd.merge(SciDTB, corpus_properties, left_on='docname', right_on='index')
    return new


###########################
#   GRAPH
###########################

# GRAPH PARSING + PRINTING

def json2dtf(jsondata):
    '''
    get json as pandas
    '''
    dtf = pd.read_json(jsondata)
    return dtf


def get_graph(hdrs):
    """
     creates a nxgraph from a list of hdrs
    """
    graph = nx.DiGraph()
    for (src, trg, rel) in hdrs:
        src = src-1
        trg = trg-1
        if rel != "ROOT":
            graph.add_edge(src, trg, label=rel)
    return graph


def draw_graph(nxgraph):
    '''
    from a nx graph, return image to be displayed in jupyter notebook
    '''
    graphimg = nx.nx_agraph.to_agraph(nxgraph)     # interface vers pygraphviz
    graphimg.layout()
    return display(Image(graphimg.draw(format='png')))

# GRAPH PROPERTIES

# create dic of node properties
def get_nodes_properties(graph):
    """
        from a graph, get a dict {edu : (inrel, outrel)}
    """
    dic_nodes_properties = {}
    for node in graph.nodes():
        dic_nodes_properties[node] = {}

        in_rels = []
        out_rels = []

        for nodetrg in graph.successors(node):
            out_rels.append(graph[node][nodetrg]["label"])
        for nodesrc in graph.predecessors(node):
            in_rels.append(graph[nodesrc][node]["label"])

        dic_nodes_properties[node]["inrels"] = in_rels
        dic_nodes_properties[node]["outrels"] = out_rels
    return dic_nodes_properties


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


def get_root_in_degree(graph):
    """
    returns the node degree of root node (ie principal argument)
    """
    root_node = get_root(graph)
    return graph.in_degree(root_node)


def get_min_in_degree(graph):
    """
    returns the mini node in deegree
    """
    in_degs = [graph.in_degree(n) for n in graph]
    # remove 0 and 1 because common to all graphs
    in_degs = [x for x in in_degs if x != 0 and x != 1]
    return min(in_degs)


def get_max_in_degree(graph):
    """
    returns the maxi node in deegree
    """
    in_degs = [graph.in_degree(n) for n in graph]
    return max(in_degs)


def get_longest_path(graph):
    """
        returns the longst path
    """
    return nx.dag_longest_path_length(graph)


# DOC DESCRIPTION

def rebuild_txt_clean(edus_txt):
    # remove specific car
    edus = [x.replace("\r", "") for x in edus_txt if x != "ROOT"]
    txt = " ".join(edus)
    txt = txt.replace("<S>", "")
    return txt


def rebuild_txt(edus_txt):
    # remove specific car
    edus = [x.replace("\r", "") for x in edus_txt if x != "ROOT"]
    txt = " ".join(edus)
    txt = txt.replace("<S>", "")
    return txt


def show_txt(dtf, docid):
    print("=== DESCR ===")
    authors = dtf.loc[docid, "authors"]
    title = dtf.loc[docid, "title"]
    proceedings = dtf.loc[docid, "proceedings"]
    publiyear = dtf.loc[docid, "publidate"].split("/")[0]
    publimonth = dtf.loc[docid, "publidate"].split("/")[1]
    print(title)
    print(authors)
    print(proceedings)
    print(str(publimonth)+"/"+str(publiyear))
    # print(publimonth)
    print("\n=== edus ===")
    for idx, edu in enumerate(dtf.loc[docid, "edustxt"][1:]):
        print(str(idx+1) + " " + edu)
    print("\n=== annot ===")
    # graph = dtf.loc[docid, "graph"]
    draw_graph(get_annot(dtf, docid))


def get_annot(dtf, docid):
    """
        return graphs by requesting the dataframe
    """
    return dtf.loc[docid, "graph"]



trantab = str.maketrans("", "", string.punctuation)
sw_all = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]


def stem(tokens, stemmer):
    stemmed = [stemmer.stem(tok) for tok in tokens]
    return stemmed


def remove_sw_en(tokens):
    """
    remove stopwords for a list of tokens
    """
    sw = stopwords.words('english')+sw_all
    filtered = [tok for tok in tokens if tok not in sw_all]
    return filtered


def segment_and_tokenize(string):
    # Sentence splitting
    sentences = sent_tokenize(string)

    # tokenizing
    content = []
    for sent in sentences:
        for w in word_tokenize(sent):
            content.append(w)
    return content


# Define a function which segments, tokenizes and removes punctuation signs
def seg_tok_nopunc(txt):
    trantab = str.maketrans("", "", string.punctuation)
    txtnopunc = txt.translate(trantab)
    tokens = segment_and_tokenize(txtnopunc)
    return tokens


def get_units(corpus, cluster, outputfile):
    """
        from text cluster, recover units and save to output file
    """
    units = []

    # for all docs
    for (text, docid) in cluster.texts:
        # for all units in text
        for eduid, edu in enumerate(corpus.loc[docid, "edustxt"]):
            units.append(edu+"\n")

    # save, one unit per line
    with open(outputfile) as output:
        output.writelines(units)


def get_coarse_rels(rels):
    """
     Todo : simplif
    """
    out = []
    for rel in rels:
        if "-" in rel:
            new = rel.split("-")[0]
            out.append(new)
        else:
            out.append(rel)
    return out


def get_root_nodes(rels_per_doc):
    """
        recover root nodes for all docs
    """
    root_nodes = []
    # loop over docs rels_sets
    for rels_set in rels_per_doc:
        # loop over rels in a doc
        for rel in rels_set:
            if rel[2] == "ROOT":
                root_nodes.append(rel[0])  # rel[0] is root node
    return root_nodes


def get_rels_per_level(graph):

    root = get_root(graph)

    # recover edges list in bfs order
    bfsedges = nx.bfs_edges(graph, root, reverse=True)

    rel_per_level = {0:[]}  # save rel per level
    levels_done = {-1:[root], 0:[]}  # save nodes per level
    cur_level = 0

    # loop over edges and save rel at each level
    for (trg, src) in bfsedges:
            rel = graph.get_edge_data(src, trg)["label"]
            if trg in levels_done[cur_level-1]:
                rel_per_level[cur_level].append(rel)
                levels_done[cur_level].append(src)
            else:
                cur_level += 1
                rel_per_level[cur_level] = [rel]
                levels_done[cur_level] = [src]

    return rel_per_level


def root_rels(hdrs, rootnode):
    """
        recover root relations
    """
    root_rels = []
    # loop over rels set for each doc
    for (rels, root) in zip(hdrs, rootnode):
        cur_root_rels = []
        # loop each rel
        for rel in rels:
            # if target is root node
            if rel[1] == root:
                # add relation to current list
                cur_root_rels.append(rel[2])
        root_rels.append(cur_root_rels)
    return root_rels


def split_annot(graph, edustxt, method="bg_el_ev"):
    """
    ATENTION ORDRE DDES LABELS DES UNITES =/= de lordre dans le fichier !!!
    ici commence à 1 : todo = change
    ToDo : généraliser à nimp quel nb de moves
    => test 2 move : eval, elab ou bg, elab
    """
    out = {"bg": [],
           "el": [],
           "ev": [],
           "autres": []
          }
    # recover root node and add to "elab" part
    root_node = get_root(graph)
    out["el"].append(root_node)

    # get root neighbors
    root_out_edges = list([x for x in graph.in_edges(root_node, data=True)])
    for out_edge in root_out_edges:
        src = out_edge[0]
        trg = out_edge[1]
        cur_rel = out_edge[2]["label"]  # recov rel lab
        if "-" in cur_rel:  # 2 coarse grain if fine grain
            cur_rel = cur_rel.split("-")[0]

        # recover sons of cur out edge
        sons = [x for x in nx.bfs_edges(graph, reverse=True, source=src)]
        sons = list(set([i for sub in sons for i in sub]))
        if sons == []:
            sons.append(src)

        # add sons to dict -> corresponding key
        if cur_rel == "bg":
            out["bg"] += sons

        if cur_rel == "evaluation":
            out["ev"] += sons

        if cur_rel == "elab":
            out['el'] += sons
    return out
