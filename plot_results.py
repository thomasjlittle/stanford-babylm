import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('results.csv')

plt.plot(results['Checkpoint'], results['perplexity'])
plt.title('Perplexity')
plt.xlabel('Model Checkpoints')
plt.ylabel('Perplexity')
plt.savefig('Perplexity.png')

plt.figure()
plt.plot(results['Checkpoint'], results['tscore0_babylm'])
plt.plot(results['Checkpoint'], results['tscore1_babylm'])
plt.plot(results['Checkpoint'], results['tscore2_babylm'])
plt.plot(results['Checkpoint'], results['tscore3_babylm'])
plt.title('Tscore - BabyLM Dataset')
plt.xlabel('Model Checkpoints')
plt.ylabel('Tscore')
plt.legend(['st_threshold = 0', 'st_threshold = 1', 'st_threshold = 2', 'st_threshold = 3'])
plt.savefig('Tscore-babylm.png')

plt.figure()
plt.plot(results['Checkpoint'], results['tscore0_cogs'])
plt.plot(results['Checkpoint'], results['tscore1_cogs'])
plt.plot(results['Checkpoint'], results['tscore2_cogs'])
plt.plot(results['Checkpoint'], results['tscore3_cogs'])
plt.title('Tscore - COGS Dataset')
plt.xlabel('Model Checkpoints')
plt.ylabel('Tscore')
plt.legend(['st_threshold = 0', 'st_threshold = 1', 'st_threshold = 2', 'st_threshold = 3'])
plt.savefig('Tscore-cogs.png')

plt.figure()
plt.plot(results['Checkpoint'], results['blimp_overall'])
plt.title('BLiMP Scores')
plt.xlabel('Model Checkpoints')
plt.ylabel('BLiMP Overall Score')
plt.savefig('BlimpScore.png')