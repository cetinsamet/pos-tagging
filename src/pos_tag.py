import sys
from nltk.tokenize import word_tokenize
from train import pos_tagger


def main(argv):
    if len(argv)!=1:
        print("Usage: python3 pos_tag.py input-sentence")
        exit()

    sent    = argv[0]
    sent    = word_tokenize(sent)

    LOAD_PATH   = '../model/pos_tagger.gz'
    tagger  = pos_tagger()
    tagger.load(LOAD_PATH)
    print(tagger.tag(sent))
    return

if __name__ == '__main__':
    main(sys.argv[1:])
