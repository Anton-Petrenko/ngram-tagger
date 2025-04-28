import csv
import json

class Tagger:
    def __init__(self):
        self.data = {}
    
    def run_jackknife(self, testing_set_size: int = 1000, iterations: int = None, highest_gram: int = 4, debug: bool = False):
        file_lines = self.get_lines_from_file()
        iter_num = len(file_lines) // testing_set_size if iterations == None else iterations
        cur_iter = 0
        while iter_num != 0:
            start = testing_set_size*cur_iter
            end = start + testing_set_size

            test_set = file_lines[start:end]
            training_set = file_lines[:start] + file_lines[end:]
            print("iter", start, end)

            self.create_model(training_set, highest_gram)
            self.save_model()
            self.evaluate_model(test_set, highest_gram, debug)

            iter_num -= 1
            cur_iter += 1

    def evaluate_model(self, lines: list[list[str]], highest_gram: int, debug: bool):
        history = []
        correct = [0]*highest_gram
        totals = [0]*highest_gram
        for line in lines:
            guesses = [self.model_guess(line[1], [], debug)]
            for i in range(highest_gram-1):
                guesses.append(self.model_guess(line[1], history[(-i-1):], debug))
            if debug: print(guesses)
            for i, guess in enumerate(guesses):
                if debug: print(f"Model {i} guesses {guess} --> {line[2]}")
                if guess == line[2]:
                    correct[i] += 1
                totals[i] += 1
            if line[2] == '.': history.clear()
            else: history.append(line[2])
            if len(history) == highest_gram: history.pop(0)
            if debug: input()
        print(correct, totals)

    def model_guess(self, word: str, history: list[str], debug: bool):
        if debug: print(f"Analyzing {word} with history {history}")
        if len(history) == 0:
            if self.data.get(word, False):
                return max(self.data[word], key=self.data[word].get)
            else:
                return "NN"
        else:
            for i in range(len(history), 0, -1):
                key = f"{'^'.join(history[-i:])}^{word}"
                if debug: print(key)
                if self.data.get(key, False):
                    if debug: print("found")
                    return max(self.data[key], key=self.data[key].get)
            if self.data.get(word, False):
                return max(self.data[word], key=self.data[word].get)
            else:
                return "NN"
                    
    def create_model(self, lines: list[list[str]], highest_gram: int = 4):
        self.create_unigram(lines)
        self.create_models(lines, highest_gram)
    
    def create_unigram(self, lines: list[list[str]]):
        for line in lines:
            self.add_entry(line[1], line[2])
    
    def create_models(self, lines: list[list[str]], highest_gram: int):
        history: list[str] = []
        for line in lines:
            if len(history) > 0 and len(self.data[line[1]].keys()) > 1:
                for i in range(len(history)):
                    self.add_entry(f"{'^'.join(history[-(i+1):])}^{line[1]}", line[2])
            if line[2] == '.': history.clear()
            else: history.append(line[2])
            if len(history) == highest_gram: history.pop(0)

    def add_entry(self, key: str, pos: str):
        assert isinstance(key, str)
        assert isinstance(pos, str)
        if self.data.get(key, False):
            if self.data[key].get(pos, False):
                self.data[key][pos] += 1
            else:
                self.data[key][pos] = 1
        else:
            self.data[key] = { pos: 1 }

    def get_lines_from_file(self):
        csv_file = open("ner_dataset2.csv", "r")
        return list(csv.reader(csv_file))[1:]

    def save_model(self, filename = "model.json"):
        with open(filename, "w") as file:
            json.dump(self.data, file)