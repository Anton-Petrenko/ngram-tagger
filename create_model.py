import pandas as pd
import sys
import os
import json

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Invalid arguments: please provide a file name to read from.', file=sys.stderr)
        sys.exit(1)

    raw_dataset = pd.read_csv(sys.argv[1])
    raw_dataset['Sentence Start'] = ~raw_dataset['Sentence #'].isna()

    # TODO: decide on start and end delimiters
    tuple_start_delimiter = '<START>'
    tuple_sep_delimiter = '<SEP>'
    tuple_end_delimiter = '<END>'

    def create_tuple_string_for_json(tup):
        return f'{tuple_start_delimiter + tuple_sep_delimiter.join(tup) + tuple_end_delimiter}'

    # TODO: see note above about deciding on delimiters
    def create_tuple_from_delimited_string(string):
        return (*string.replace(tuple_start_delimiter, '').replace(tuple_end_delimiter, '').split(tuple_sep_delimiter),)

    # index 0 is unigram, 1 is bigram, 2 is trigram, 3 is quadgram
    # the keys of each dictionary are effectively a path to the occurence
    #   for a unigram, this would be simply the current word
    #   for higher n-grams, this would be the sequence of previous POS with the current word
    # the values of each dictionary are also dictionaries
    #   the key is the observed type
    #   the value is the number of times this POS occurred
    ngram_maps = [{}, {}, {}, {}]

    previously_seen = []
    for index, row in raw_dataset.iterrows():
        # reset to having seen nothing if new sentence starts
        if row['Sentence Start']:
            previously_seen = []
        
        # unigram
        if row['Word'] not in ngram_maps[0]:
            ngram_maps[0][row['Word']] = { row['POS']: 1 }
        else:
            if row['POS'] in ngram_maps[0][row['Word']]:
                ngram_maps[0][row['Word']][row['POS']] += 1
            else:
                ngram_maps[0][row['Word']][row['POS']] = 1

        # higher n-grams
        for n in range(1, len(ngram_maps)):
            # skip higher grams if not enough previously seen words in this sentence
            if len(previously_seen) < n:
                break

            # unpacks the last n elements into a tuple with word, then check if that pattern has been observed yet
            pattern = (*previously_seen[-n:len(previously_seen)], row['Word'])
            pattern = create_tuple_string_for_json(pattern)
            if pattern not in ngram_maps[n]:
                ngram_maps[n][pattern] = { row['POS']: 1 }
            else:
                if row['POS'] in ngram_maps[n][pattern]:
                    ngram_maps[n][pattern][row['POS']] += 1
                else:
                    ngram_maps[n][pattern][row['POS']] = 1

        # store the part of speech
        previously_seen.append(row['POS'])

    # generate model as json
    if not os.path.isdir('models'):
        os.mkdir('models')
    with open(f'models/{sys.argv[2] if len(sys.argv) > 2 else 'model.json'}', 'w') as json_file:
        json.dump(ngram_maps, json_file, indent=4)