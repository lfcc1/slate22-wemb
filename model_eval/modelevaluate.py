from gensim.models import KeyedVectors
from pprint import pprint
from getopt import getopt
import sys



ops,args = getopt(sys.argv[1:],"aw")
ops = dict(ops)

if len(args) < 2 :
    print("Please insert 2 arguments: [modelPath] [testFilePath]")
    exit()



if "-a" in ops:
    model = KeyedVectors.load(args[0])
    sections = model.evaluate_word_analogies(args[1])
    res = []
    for section in sections[1]:
        total_length = len(section["correct"]) + len(section["incorrect"])
        if total_length != 0:
        	section_score = len(section["correct"]) / total_length
        	res.append({"section":section["section"], "score": round(section_score*100,2), "size": total_length})
    		pprint(res)
    

elif "-w" in ops:
    model = KeyedVectors.load(args[0])
    print(model.evaluate_word_pairs(args[1]))

else:
    print("""Please select an evaluation option:
            -a : Evaluate model with analogies file
            -w : Evaluate model with pair of words similarity
        """)
