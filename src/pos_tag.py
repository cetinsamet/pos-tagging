import sys
from nltk.tokenize import word_tokenize
from train import PosTagger


def main(argv):
    if len(argv)!=1:
        print("Usage: python3 pos_tag.py input-sentence")
        exit()

    # READ USER INPUT SENTENCE
    sent        = argv[0]

    # TOKENIZE INPUT SENTENCE
    sent        = word_tokenize(sent)

    # LOAD TRAINED POS TAGGER
    LOAD_PATH   = '../model/postag_model.gz'
    tagger      = PosTagger()
    tagger.load(LOAD_PATH)

    # DISPLAY TAGGED SENTENCE
    print(tagger.tag(sent))

    return

if __name__ == '__main__':
    main(sys.argv[1:])
