import json
import pandas as pd
def train_data():
    dt = {}
    raw_dataset = pd.read_csv('ner_dataset2.csv')
    q = []
    count = 0
    for x in range(len(raw_dataset['Sentence #'])):
        if isinstance(raw_dataset['Sentence #'][x],str):
            dt[count] = q
            q = []
            count += 1
        if str(raw_dataset['Word'][x]) == 'nan':
            if isinstance(raw_dataset['Sentence #'][x],str):
                q.append(('None',raw_dataset['POS'][x]))
            else:
                q.append(('none',raw_dataset['POS'][x]))
        else:
            q.append((str(raw_dataset['Word'][x]),raw_dataset['POS'][x]))
    dt[count] = q
    
    with open('sentences.json','w') as json_file:
        json.dump(dt,json_file,indent=4)
    
    return
if __name__ == '__main__':
    train_data()