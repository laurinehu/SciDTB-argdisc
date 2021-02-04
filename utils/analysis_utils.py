nlp_terms = ["semantic parsing", "semantics parser", ""]


def doc_having_term(nlp_term: str, corpus: list):
    """
        return list of docs having a spec nlp term
    """
    print(nlp_term)
    docs = []
    for doc in corpus:
        if nlp_term.lower() in doc.lower():
            docs.append(doc)
    return docs


def test_nlp_terms(corpus):
    """
        prints prop of nlp terms
    """
    # recover docs
    docs = [x[1]["txt"] for x in corpus.corpus.iterrows()]
    
    # representation
    print(len(doc_having_term("bag of words", docs)))
    print(len(doc_having_term("bow", docs)))
    print(len(doc_having_term("embedding", docs)))
    print(len(doc_having_term("feature", docs)))
    print(len(doc_having_term("feature", docs)))
    print(len(doc_having_term("vector", docs)))

    # method of analysis
    print(len(doc_having_term("cluster", docs)))
    print(len(doc_having_term("mining", docs)))
    print(len(doc_having_term("extraction", docs)))
    print(len(doc_having_term("parsing", docs)))
    print(len(doc_having_term("parser", docs)))
    print(len(doc_having_term("translat", docs)))
    print(len(doc_having_term("learning", docs)))
    print(len(doc_having_term("deep", docs)))


    # linguistic object
    print(len(doc_having_term("named entities", docs)))
    print(len(doc_having_term("named entity", docs)))
    print(len(doc_having_term("corpus", docs)))
    print(len(doc_having_term("word", docs)))
    print(len(doc_having_term("word sense", docs)))
    print(len(doc_having_term("lexical", docs)))
    print(len(doc_having_term("semantic", docs)))
    print(len(doc_having_term("syntactic", docs)))
    print(len(doc_having_term("morpho", docs)))
    print(len(doc_having_term("sentiment", docs)))
    print(len(doc_having_term("opinion", docs)))
