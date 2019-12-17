import pickle

with open('raw.pickle', 'rb') as f:
    content = pickle.load(f)
    print(content)
