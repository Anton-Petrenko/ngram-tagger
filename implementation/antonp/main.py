import sys
from lib import Tagger

highest_gram = int(sys.argv[1])
assert 1 <= highest_gram <= 4

tagger = Tagger()
tagger.run_jackknife(1000, None, highest_gram)