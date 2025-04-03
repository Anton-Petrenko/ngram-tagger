import json
import sys
if __name__ == '__main__':
    sentence = sys.argv[1]
    words = sentence.split(' ')
    data = None
    with open(r'./models/model.json','r') as file:
        data = json.load(file)
    res = []
    q = [None,None] 

    for word in words:
        q[0],q[1] = q[1] if len(res) == 0 else res[-1],word #Treat like a queue where q[0]=POS and q[1]=literal word
        max_count = 0
        max_pos = None
        if q[0] == None: #Edge case if we're looking at only the first word, default to unigram tagger
            if word in data[0]:
                for pos in data[0][word]:
                    if data[0][word][pos] > max_count:
                        max_count = data[0][word][pos]
                        max_pos = pos
        else:
            key = f'<START>{q[0]}<SEP>{word}<END>'
            if key in data[1]: #Same process as unigram tagger, find max POS
                for pos in data[1][key]:
                    if data[1][key][pos] > max_count:
                        max_pos = pos
                        max_count = data[1][key][pos]
        res.append(max_pos if max_pos is not None else 'NN') #Default to NN if key is not found
    print(res)
    #TODO: Implement splitting the data into training and testing sets