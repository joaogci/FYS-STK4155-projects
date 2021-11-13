import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

inputs = cancer_data.data
target = cancer_data.target

data_frame = pd.DataFrame(inputs, columns=cancer_data.feature_names)

fig, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data=data_frame.corr(), annot=True, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title("Covariance Matrix for the Features of the Breast Cancer Dataset")
plt.subplots_adjust(left=0.11,
                    bottom=0.1, 
                    right=1, 
                    top=0.98, 
                    wspace=0.3, 
                    hspace=0.6)

plt.savefig("./figs/corr_cancer.pdf", dpi=400)
# plt.show()




