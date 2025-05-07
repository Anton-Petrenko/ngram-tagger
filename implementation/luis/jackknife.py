import json
import sys
def train_model(tset):
    model = [ {} for x in range(4)]
    multimodel = [{} for x in range(4)]
    unigram = model[0]
    bigrams = model[1]
    trigrams = model[2]
    quadgrams = model[3] 

    for sentence in tset:
        bigram = [None,None]
        trigram = [None,None,None]
        quadgram = [None,None,None,None]
        prev = None
        for word in sentence:
            if word[0] not in unigram:
                unigram[word[0]] = {}
            if word[1] not in unigram[word[0]]:
                unigram[word[0]][word[1]] = 0   
            unigram[word[0]][word[1]] += 1
            

            bigram[0],bigram[1] = prev,word[0]
            trigram[0],trigram[1],trigram[2] = trigram[1],prev,word[0]
            quadgram[0],quadgram[1],quadgram[2],quadgram[3] = quadgram[1],quadgram[2],prev,word[0]
            prev = word[1]
            if tuple(bigram) not in bigrams:
                bigrams[tuple(bigram)] = {}
            if word[1] not in bigrams[tuple(bigram)]:
                bigrams[tuple(bigram)][word[1]] = 0
            bigrams[tuple(bigram)][word[1]] += 1
            
            if tuple(trigram) not in trigrams:
                trigrams[tuple(trigram)] = {}
            if word[1] not in trigrams[tuple(trigram)]:
                trigrams[tuple(trigram)][word[1]] = 0
            trigrams[tuple(trigram)][word[1]] += 1
            
            if tuple(quadgram) not in quadgrams:
                quadgrams[tuple(quadgram)] = {}
            if word[1] not in quadgrams[tuple(quadgram)]:
                quadgrams[tuple(quadgram)][word[1]] = 0
            quadgrams[tuple(quadgram)][word[1]] += 1

    for x in range(4):
        mdl = model[x]
        for key in list(mdl.keys()):
            if len(mdl[key]) > 1:
                maxvalue = max(mdl[key].values())
                maxval = set()
                for y in mdl[key]:
                    if mdl[key][y] == maxvalue:
                        maxval.add(y)
                multimodel[x][key] = maxval
                del mdl[key]
            else:
                pos = [x for x in model[x][key]]
                model[x][key] = pos[0]
    return model + multimodel
def unigram_model(testing,model):
    res = 0
    words = 0
    for sentence in testing:
        words += len(sentence)
        for word,pos in sentence:
            if word in model[0]:
                if model[0][word] == pos:
                    res += 1
                    continue
            if word in model[4]:
                if pos in model[4][word]:
                    res += 1
                    continue
            if pos == 'NN':
                res += 1
                continue
    print('unigram:',res/words)
    return res/words
def bigram_model(testing,model):
    res = 0
    words = 0
    for sentence in testing:
        bi = [None,None]
        prev = None
        words += len(sentence)
        for word,pos in sentence:
            bi[0],bi[1] = prev,word
            prev = pos
            if word in model[0]:
                if model[0][word] == pos:
                    res += 1
                    continue
            if tuple(bi) in model[1]:
                if model[1][tuple(bi)] == pos:
                    res += 1
                    continue
            if tuple(bi) in model[5]:
                if pos in model[5][tuple(bi)]:
                    res += 1
                    continue
            if word in model[4]:
                if pos in model[4][word]:
                    res += 1
                    continue
                if pos == 'NN':
                    res += 1
                    continue
    print('bigram:',res/words)
    return res/words
def trigram_model(testing,model):
    res = 0
    words = 0
    
    for sentence in testing:
        bi = [None,None]
        tri = [None,None,None]
        prev = None
        words += len(sentence)
        for word,pos in sentence:
            bi[0],bi[1] = prev,word
            tri[0],tri[1],tri[2] = tri[1],prev,word
            prev = pos
            if word in model[0]:
                if model[0][word] == pos:
                    res += 1
                    continue
            if tuple(bi) in model[1]:
                if model[1][tuple(bi)] == pos:
                    res += 1
                    continue
            if tuple(tri) in model[2]:
                if model[2][tuple(tri)] == pos:
                    res += 1
                    continue
            if tuple(tri) in model[6]:
                if pos in model[6][tuple(tri)]:
                    res += 1
                    continue
            if tuple(bi) in model[5]:
                if pos in model[5][tuple(bi)]:
                    res += 1
                    continue
            if word in model[4]:
                if pos in model[4][word]:
                    res += 1
                    continue
                if pos == 'NN':
                    res += 1
                    continue
    print('trigram:',res/words)
    return res/words

def quadgram_model(testing,model):
    res = 0
    words = 0
    for sentence in testing:
        bi = [None,None]
        tri = [None,None,None]
        quad = [None,None,None,None]
        prev = None
        words += len(sentence)
        for word,pos in sentence:
            bi[0],bi[1] = prev,word
            tri[0],tri[1],tri[2] = tri[1],prev,word
            quad[0],quad[1],quad[2],quad[3] = quad[1],quad[2],prev,word
            prev = pos
            temp = res
            if word in model[0]:
                if model[0][word] == pos:
                    res += 1
                    continue
            if tuple(bi) in model[1]:
                if model[1][tuple(bi)] == pos:
                    res += 1
                    continue
            if tuple(tri) in model[2]:
                if model[2][tuple(tri)] == pos:
                    res += 1
                    continue
            if tuple(quad) in model[3]:
                if model[3][tuple(quad)] == pos:
                    res += 1
                    continue
            if tuple(quad) in model[7]:
                if pos in model[7][tuple(quad)]:
                    res += 1
                    continue
            if tuple(tri) in model[6]:
                if pos in model[6][tuple(tri)]:
                    res += 1
                    continue
            if tuple(bi) in model[5]:
                if pos in model[5][tuple(bi)]:
                    res += 1
                    continue
            if word in model[4]:
                if pos in model[4][word]:
                    res += 1
                    continue
            if pos == 'NN':
                res += 1
                continue
    print('quadgram:',res/words)
    return res/words

if __name__ == '__main__':
    
    with open('sentences.json','r',encoding='utf-8') as f:
        inp = int(sys.argv[1])
        sentences = json.load(f)
        uni = 0
        bi = 0
        tri = 0
        quad = 0
        count = 0 

        for x in sentences:
            count += len(sentences[x])
        
        total_set = 0
        for x in range(1,len(sentences),1000):
            testing_set = []
            training_set = []
            
            for y in range(1,x):
                sentence = []
                for word in sentences[str(y)]:
                    sentence.append(word)
                training_set.append(sentence)

            for y in range(x,x+1000):
                if y >= len(sentences):
                    
                    break
                sentence = []
                for word in sentences[str(y)]:
                    sentence.append(word)
                testing_set.append(sentence)
            

            for y in range(x+1000,len(sentences)):
                sentence = []
                for word in sentences[str(y)]:
                    sentence.append(word)
                training_set.append(sentence)
            model = train_model(training_set)
            if inp == 1:
                uni += unigram_model(testing_set,model)
            elif inp == 2:
                uni += unigram_model(testing_set,model)
                bi += bigram_model(testing_set,model)
            elif inp == 3:
                uni += unigram_model(testing_set,model)
                bi += bigram_model(testing_set,model)
                tri += trigram_model(testing_set,model)
            elif inp == 4:
                uni += unigram_model(testing_set,model)
                bi += bigram_model(testing_set,model)
                tri += trigram_model(testing_set,model)
                quad += quadgram_model(testing_set,model)
        print("final unigram:",uni/48)
        print("final bigram:",bi/48)
        print("final trigram:",tri/48)
        print("final quadgram:",quad/48)