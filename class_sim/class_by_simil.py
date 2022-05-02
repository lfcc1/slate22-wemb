from gensim.models import KeyedVectors
import yaml
from getopt import getopt
import sys
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ops,args = getopt(sys.argv[1:],"vuhcpt")
ops = dict(ops)


model = KeyedVectors.load(args[0])

f = open(args[1], encoding="utf8")
sections= yaml.safe_load(f)


score = 0
oov = []
length = 0





def display_scatterplot(model, sections,pca=True):
    words = [test[0]  for sec in sections for test in sec["testes"] if test[0] in model.key_to_index]
    print(words)
    word_vectors = np.array([model[w] for w in words])

    if pca:
        twodim = PCA().fit_transform(word_vectors)[:,:2]
    else:
        twodim = TSNE(n_components=2,random_state=0 ,perplexity=20,n_iter=10000).fit_transform(word_vectors)[:,:2]
    
    plt.figure(figsize=(6,6))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05, y+0.05, word)
    plt.show()



def getTestsLength(sections):
    count = 0
    for section in sections:
        if section["score"] != -1:
            count += len(section["testes"])
    return count

def createVerboseResult(sections):
    result = []
    for section in sections:
        if section["score"] == -1:
            continue

        accepted = []
        rejected = []
        for test in section["testes"]:
            try:
                accepted.append(test[0]) if test[2] == 1 else rejected.append(test[0])
            except IndexError:
                continue
        result.append({"section":section["tit"],"score":section["score"],"accepted":accepted,"rejected":rejected})
    return result

def isUnknownWord(word):
    if word not in model.key_to_index:
        oov.append(word)
        return True 
    return False

def containsUnknownWord(words):
    for word in words:
        if isUnknownWord(word):
            return True
    return False


for section in sections:
    section["score"] = -1
    if containsUnknownWord(section["clas"]):
            continue
    local_score = 0
    local_length = 0
    clusters = {clas:[] for clas in section["clas"]}
    for test in section['testes']:
        if isUnknownWord(test[0]):
            continue
        
        local_length += 1
        sim_result = [model.similarity(test[0],clas) for clas in section['clas']]
        found_index = sim_result.index(max(sim_result))

        #print(index,teste[0],elem['clas'][index])
        clusters[section['clas'][found_index]].append(test[0])
        if found_index == test[1] or section['clas'][found_index] == test[1]:
            local_score += 1
            test.append(1)  ## Acepted
        else:
            test.append(0) ## Rejected
    section["clusters"] = clusters
    section["score"] = round(local_score/(local_length),4) if local_length != 0 else 0
    score += local_score
    length += local_length

total_score = round(score/(length),4) if length != 0 else 0

real_length = getTestsLength(sections)
oov_ratio = round(len(oov)/real_length,4) if real_length != 0 else 0

if '-v' in ops:
    result = ({"global_score":total_score,"oov_ratio":oov_ratio}, createVerboseResult(sections))
    pprint(result)

elif '-u' in ops:
    print((oov_ratio,oov))

elif '-c' in ops:
    result = [(section["tit"],section["clusters"]) for section in sections]
    pprint(result)
elif '-p' in ops:
    display_scatterplot(model,sections)

elif '-t' in ops:
    display_scatterplot(model,sections,False)

elif "-h" in ops: 
    print("""Usage:
            -u : Unknown words list and ratio.
            -v : Verbose, list accepted and rejected tokens for each section
            -c : Lists all tokens associated with predicted class.
            -p : Display plot with word vectors (PCA).
            -t : Display plot with word vectors (T-SNE).
        """)

else:
    result = ({"global_score":total_score,"oov_ratio":oov_ratio}, [(section["tit"],section["score"]) for section in sections])     
    print(result)
