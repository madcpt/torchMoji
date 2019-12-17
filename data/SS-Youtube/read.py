import pickle
import numpy

with open('raw.pickle', 'rb') as f:
    content = pickle.load(f)

print(content.keys())
print(content['texts'][0])
print(content['info'][0])
print(len([i for i in content['info'] if i['label']==1]))
