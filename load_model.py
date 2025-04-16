import pandas as pd
import sys
import os
import json

# TODO: decide on start and end delimiters
tuple_start_delimiter = '<START>'
tuple_sep_delimiter = '<SEP>'
tuple_end_delimiter = '<END>'

def create_tuple_from_delimited_string(string):
    return (*string.replace(tuple_start_delimiter, '').replace(tuple_end_delimiter, '').split(tuple_sep_delimiter),)

def load_model_as_tuples(filepath):
    with open(filepath, 'r') as json_file:
        raw = json.load(json_file)

    data = []

    for set in raw[1:]:
        new_set = {}
        for old_key in set.keys():
            new_set[create_tuple_from_delimited_string(old_key)] = set[old_key]
        data.append(new_set)

    return data

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Invalid arguments: please provide a file name to read from.', file=sys.stderr)
        sys.exit(1)
    
    model = load_model_as_tuples(sys.argv[1])
    print(model)