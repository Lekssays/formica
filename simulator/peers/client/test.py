import numpy as np 

weights = []
THRESHOLD = 0.50

chosen_weights = [{"messageID": "ms1", "client": 1}, {"messageID": "ms2", "client": 2}]
trust_score = np.array([0.4,0.2,0.9])
similarity = np.array([[0,0.8,0.5], [0.8,0,0.3], [0.5,0.3,0]])

metrics = []
for i in range(0, len(chosen_weights)):
    tmp = {
        'trust_score': trust_score[chosen_weights[i]['client']],
        'similarity': similarity[chosen_weights[i]['client']][0],
        'messageID': chosen_weights[i]['messageID']
    }
    metrics.append(tmp)

metrics = sorted(metrics, key=lambda x: x['similarity'], reverse=True)

for m in metrics:
    if 1 - m['trust_score'] > THRESHOLD:
        metrics.remove(m)

print(metrics)