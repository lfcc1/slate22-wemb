from gensim.models import KeyedVectors
from pprint import pprint
from getopt import getopt
import sys



ops,args = getopt(sys.argv[1:],"avw")
ops = dict(ops)

if len(args) < 2 :
    print("Please insert 2 arguments: [modelPath] [testFilePath]")
    exit()



if "-a" in ops:
    model = KeyedVectors.load(args[0])
    sections = model.wv.evaluate_word_analogies(args[1])
    res = []
    for section in sections[1]:
        total_length = len(section["correct"]) + len(section["incorrect"])
        section_score = len(section["correct"]) / total_length
        res.append({"section":section["section"], "score": round(section_score,4), "size": total_length})
    pprint(res)
    

elif "-w" in ops:
    model = KeyedVectors.load(args[0])
    print(model.wv.evaluate_word_pairs(args[1]))

else:
    print("""Please select evaluation option:
            -a : Evaluate model with analogies file
            -w : Evaluate model with pairs of words similarity
        """)