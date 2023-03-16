import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('results.csv')

plt.plot(results['Checkpoint'], results['perplexity'])
plt.title('Perplexity')
plt.xlabel('Model Checkpoints')
plt.ylabel('Perplexity')
plt.savefig('Perplexity.png')

plt.figure()
plt.plot(results['Checkpoint'], results['tscore0'])
plt.plot(results['Checkpoint'], results['tscore1'])
plt.plot(results['Checkpoint'], results['tscore2'])
plt.plot(results['Checkpoint'], results['tscore3'])
plt.title('Tscore')
plt.xlabel('Model Checkpoints')
plt.ylabel('Tscore')
plt.legend(['st_threshold = 0', 'st_threshold = 1', 'st_threshold = 2', 'st_threshold = 3'])
plt.savefig('Tscore.png')

plt.figure()
plt.plot(results['Checkpoint'], results['blimp_overall'])
plt.title('BLiMP Scores')
plt.xlabel('Model Checkpoints')
plt.ylabel('BLiMP Overall Score')
plt.savefig('BlimpScore.png')