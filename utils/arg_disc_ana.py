def prop_sim_rels(ag, dg, direc=False, labels=False):
    agedges = ag.edges()
    dgedges = dg.edges()
    # if undirected comparison, convert list to set
    if direc is False:
        agedges = [set(e) for e in agedges]
        dgedges = [set(e) for e in dgedges]
    elif labels is True:
        agedges = ag.edges(data=True)
        dgedges = dg.edges(data=True)
    else:
        agedges = ag.edges()
        dgedges = dg.edges()

    sim = 0
    tot = len(agedges)
    for e in dgedges:
        if e in agedges:
            sim+=1
    return sim/tot
