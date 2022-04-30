from gensim.models import KeyedVectors

from getopt import getopt
import sys



ops,args = getopt(sys.argv[1:],"avw")
ops = dict(ops)

if len(args) < 2 :
    print("Please insert 2 arguments: [modelPath] [testFilePath]")
    exit()



if "-a" in ops:
    model = KeyedVectors.load(args[0])
    res = model.wv.evaluate_word_analogies(args[1])
    print(res[0])
    

elif "-w" in ops:
    model = KeyedVectors.load(args[0])
    print(model.wv.evaluate_word_pairs(args[1]))

else:
    print("""Please select evaluation option:
            -a : Evaluate model with analogies file
            -w : Evaluate model with pairs of words similarity
        """)