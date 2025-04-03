import json
import sys
if __name__ == '__main__':
    sentence = sys.argv[1]
    words = sentence.split(' ')
    data = None
    with open(r'./models/model.json','r') as file:
        data = json.load(file)
    res = []
    for word in words: #Read each word
        r = None
        max_count = 0
        if word in data[0]:
            for pos in data[0][word]: #If the word is in the dictionary, check for max POS count
                if data[0][word][pos] > max_count:
                    r = pos
                    max_count = data[0][word][pos]
        res.append(r if r is not None else "NN") #Default to noun if not found else, add the max POS
    print(res)
#TODO: Implement splitting the data into training and testing sets