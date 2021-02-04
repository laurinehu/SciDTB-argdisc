import re

def launch_predictor(sent, predictor):
    """
    return alennlp annotation
    """
    return predictor.predict(sentence=sent)

def get_args(descr):
    """
        returns list of args dict from descr str
    """
    args = {}
    for match in re.findall(r'\[([^:]*): ([^\]]*)\]',descr):
        if match[0] != "V":
            args[match[0]] = match[1]
    return args

def get_verbs(annot):
    """
        returns dict of verbs for a sent annot
    """
    out_verbs = {}
    for prop in annot:
        #print(prop)
        descr = prop["description"]
        
        out_verbs[prop["verb"].lower()] = [get_args(descr)]
    return out_verbs

def srl(sent, predictor):
    return get_verbs(launch_predictor(sent, predictor)["verbs"])