from gensim.models import KeyedVectors
import yaml
from getopt import getopt
import sys

ops,args = getopt(sys.argv[1:],"vu")
ops = dict(ops)


model = KeyedVectors.load(args[0])

f = open(args[1], encoding="utf8")
sections= yaml.safe_load(f)


score = 0
oov = []
length = 0


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
    for test in section['testes']:
        if isUnknownWord(test[0]):
            continue
        
        local_length += 1
        sim_result = [model.similarity(test[0],clas) for clas in section['clas']]
        found_index = sim_result.index(max(sim_result))

        #print(index,teste[0],elem['clas'][index])
        if found_index == test[1] or section['clas'][found_index] == test[1]:
            local_score += 1
            test.append(1)  ## Acepted
        else:
            test.append(0) ## Rejected

    section["score"] = round(local_score/(local_length),4) if local_length != 0 else 0
    score += local_score
    length += local_length

total_score = round(score/(length),4) if length != 0 else 0

real_length = getTestsLength(sections)
oov_ratio = round(len(oov)/real_length,4) if real_length != 0 else 0

if '-v' in ops:
    result = ({"global_score":total_score,"oov_ratio":oov_ratio}, createVerboseResult(sections))
    print(result)

elif '-u' in ops:
    print((oov_ratio,oov))

else:
    result = ({"global_score":total_score,"oov_ratio":oov_ratio}, [(section["tit"],section["score"]) for section in sections])     
    print(result)
