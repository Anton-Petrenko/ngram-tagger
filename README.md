# Structure

In the `report` directory, you will find the LaTeX files necessary to edit and generate the report.

In the `implementation` directory, you will find three folders for each approach implemented based on the initial framework.

***IMPORTANT***: Ensure that the dataset file (`ner_dataset2.csv`) is located in the root directory of the repository.

# Testing

For each implementation, do the following:

## Anton

1. Navigate to the root directory of the repo.

2. Run and time Python script for each desired iteration.

```bash
time python implementation/antonp/main.py 4
time python implementation/antonp/main.py 3
time python implementation/antonp/main.py 2
time python implementation/antonp/main.py 1
```

## Luis

1. Run and time Python script for each desired iteration.

```bash
time python jackknife.py 4
time python jackknife.py 3
time python jackknife.py 2
time python jackknife.py 1
```

## Merritt

The following instructions cover setup on a Windows machine. Different commands may be required on different operating systems.
Requirements: Python 3.12 or up (likely possible with older Python 3 versions but untested)

1. Navigate to the `implementation/merritt` directory.

```bash
cd implementation/merritt
```

2. Create and activate a new Python virtual enivronment.

```bash
python -m venv venv
source venv/Scripts/activate
```

3. Install required modules into venv.

```bash
pip install -r requirements.txt
```

4. Run and time Python script for each desired iteration.

```bash
time python ngram_tests.py quadgram_output.txt 4
time python ngram_tests.py trigram_output.txt 3
time python ngram_tests.py bigram_output.txt 2
time python ngram_tests.py unigram_output.txt 1
```