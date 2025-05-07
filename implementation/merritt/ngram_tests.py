import pandas as pd
import numpy as np
import sys

def generate_ngram_model(training_set, gram_size=4):
    # ---------------- STAGE 0: PREPROCESSING HISTORY ----------------
    modified = pd.DataFrame(training_set)

    for n in range(gram_size - 1):
        modified[f'Previous POS {n + 1}'] = modified['POS'].shift(n + 1)
        modified[f'Previous POS {n + 1}'] = modified[f'Previous POS {n + 1}'].where(modified['Sentence #'] == modified['Sentence #'].shift(n + 1))

    # ---------------- STAGE 1: INITIAL COUNTING ----------------

    # the keys of this dictionary are effectively a path to the occurence
    #   for a unigram, this would be simply the current word
    #   for higher n-grams, this would be the sequence of previous POS with the current word
    # the values of each dictionary are also dictionaries
    #   the key is the observed type
    #   the value is the number of times this POS occurred
    ngram_map = {}

    def write_to_map(row):
        previous_pos = list(map(lambda n : row[f'Previous POS {n}'], list(range(1, gram_size))[::-1]))
        previous_pos = list(filter(lambda pos : not pd.isna(pos), previous_pos))

        patterns = [(*previous_pos[-i:], row['Word']) for i in range(1, len(previous_pos) + 1)]
        patterns.append(row['Word'])

        for pattern in patterns:
            if pattern in ngram_map:
                if row['POS'] in ngram_map[pattern]:
                    ngram_map[pattern][row['POS']] += 1
                else:
                    ngram_map[pattern][row['POS']] = 1
            else:
                ngram_map[pattern] = { row['POS']: 1 }

    modified.apply(write_to_map, axis=1)

    # ---------------- STAGE 2: FLATTENING ----------------

    # index 0 is the unanimous map, index 1 is the highest probability map
    ngram_model = [{}, {}]
    # iterate over every key, and store it in either the unanimous or non unanimous layer
    for (key, pos_map) in ngram_map.items():
        # check if key should be added to the unanimous map or the highest probability map
        map_index = 0 if len(pos_map) == 1 else 1
        
        # only add to map if not ambiguous - this did not yield positive results
        # if list(pos_map.values()).count(pos_map[max(pos_map)]) == 1:
        #     ngram_model[map_index][key] = max(pos_map, key=pos_map.get)

        ngram_model[map_index][key] = max(pos_map, key=pos_map.get)

    return (ngram_map, ngram_model)

def test_model(testing_set, model, gram_size=4, output_log_filename='output_log.txt'):
    # ---------------- STAGE 0: PREPROCESSING HISTORY ----------------
    modified = pd.DataFrame(testing_set)

    for n in range(gram_size - 1):
        modified[f'Previous POS {n + 1}'] = modified['POS'].shift(n + 1)
        modified[f'Previous POS {n + 1}'] = modified[f'Previous POS {n + 1}'].where(modified['Sentence #'] == modified['Sentence #'].shift(n + 1))

    # ---------------- STAGE 1: TESTING EACH WORD ----------------
    def test_with_model(row):
        previous_pos = list(map(lambda n : row[f'Previous POS {n}'], list(range(1, gram_size))[::-1]))
        previous_pos = list(filter(lambda pos : not pd.isna(pos), previous_pos))
        pattern = (*previous_pos, row['Word'])
        # construct n grams
        grams = []
        for gram_index in range(0, gram_size):
            if len(pattern) >= gram_index+1:
                key = tuple(pattern[min(len(pattern), gram_size)-gram_index-1:]) if gram_index > 0 else pattern[len(pattern)-1]
                grams.append(key)

        # tesing against model
        expected = row['POS']
        answer = None
        # unanimous, in ascending order
        for gram in grams:
            if gram in model[0]:
                answer = model[0][gram]
                break 
        # ngrams, in descending order
        if answer is None:
            for gram in grams[::-1]:
                if gram in model[1]:
                    answer = model[1][gram]
                    break
        # failsafe
        if answer is None:
            answer = 'NN'

        # logging and counting results
        with open(output_log_filename, 'a', encoding='utf-8') as output_file:
            print(f'{row['Word']}\t{expected}\t{answer}\t{'MATCH' if answer == expected else 'MISMATCH'}', file=output_file)
        return answer == expected
    
    modified['Correct Guess'] = modified.apply(test_with_model, axis=1)
    return (modified['Correct Guess'].sum() / len(testing_set), modified['Correct Guess'].sum())

def jackknife_training(raw_dataset, slice_size, gram_size=4, output_log_filename='output_log.txt'):
    sentence_count = raw_dataset['Sentence #'].max()
    slice_index = 1
    total_correct = 0

    # slicing out testing sets and leaving remaining elements for training
    while slice_index < sentence_count:
        testing_set = raw_dataset[raw_dataset['Sentence #'].between(slice_index, slice_index + slice_size - 1)]
        training_set = raw_dataset[~raw_dataset['Sentence #'].between(slice_index, slice_index + slice_size - 1)]

        print(f"Training on slice ({slice_index}, {slice_index + slice_size - 1})")
        (_, model) = generate_ngram_model(training_set, gram_size)
        (accuracy, correct) = test_model(testing_set, model, gram_size, output_log_filename)
        print(f'Accuracy: {accuracy}')

        total_correct += correct
        slice_index += slice_size

    print(f'Overall accuracy: {total_correct / len(raw_dataset)}, Correct: {total_correct}, Incorrect: {len(raw_dataset) - total_correct}')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Generating quadgrams as default.')
        output_file = 'output_log.txt'
        gram_size = 4
    else:
        output_file = sys.argv[1]
        gram_size = int(sys.argv[2])

    # initial data processing
    #   cascading the sentence number down to the sentences below it
    #   necessary for dividing into training and testing sets
    raw_dataset = pd.read_csv('../../ner_dataset2.csv', na_filter=False, dtype=str)
    raw_dataset['Sentence Start'] = ~(raw_dataset['Sentence #'] == '')
    raw_dataset['Sentence #'] = raw_dataset['Sentence #'].str.extract(r'(\d+)', expand=False).ffill().astype('int64')

    # run iterations
    jackknife_training(raw_dataset, 1000, gram_size, output_file)