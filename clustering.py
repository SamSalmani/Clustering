import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

whisky = pd.read_csv("whiskies.txt")
whisky["Region"] =  pd.read_csv("regions.txt")

whisky.head()
whisky.tail()
whisky.iloc[0:10, 0:5]
whisky.columns
flavors = whisky.iloc[:, 2:14]

#Pearson Correlation
corr_flavors = pd.DataFrame.corr(flavors)
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavors.pdf")

corr_whiskies = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10,10))
plt.pcolor(corr_whiskies)
plt.axis("tight")
plt.colorbar()
plt.savefig("corr_whiskies.pdf")

#CLustering whiskies by flavor profile
# using scikit learn ML module named Spectral Co-clustering
# since the whiskies come from 6 regions, clustering algorithm needs to define 6 blocks

from sklearn.cluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters = 6, random_state = 0)
model.fit(corr_whiskies)
model.rows_
np.sum(model.rows_, axis=1)
model.row_labels_

# draw the clusters defined
whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index) #append group labels and append to whisky
whisky =  whisky.iloc[np.argsort(model.row_labels_)]# increasing order based on group label
whisky = whisky.reset_index(drop=True)#reset the index of the Dataframe

correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())#recalc correlation matrix and turn it into Numpy Array
correlations = np.array(correlations)

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whiskies)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.savefig("correlations.pdf")
