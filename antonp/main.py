from lib import Tagger

tagger = Tagger()
# lis = tagger.get_sentences_from_file()
tagger.run_jackknife(21500, None)