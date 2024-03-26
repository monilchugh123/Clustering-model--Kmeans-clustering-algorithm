#!/usr/bin/env python
# coding: utf-8

# In[154]:


import pandas as pd
import numpy as np
from PIL import Image
import random


# In[155]:


img = Image.open('assignment k means pic.jpg')


# In[156]:


w, h = img.size


# In[157]:


rgb_values = []
for y in range(h):
    for x in range(w):
        pixelvalue = img.getpixel((x,y))
        rgb_values.append(pixelvalue)


# In[158]:


df =pd.DataFrame(rgb_values, columns=['Red', 'Green','Blue'])


# In[159]:


df.shape


# In[160]:


df.head()


# In[161]:


from sklearn.cluster import KMeans

wcss = []

for i in range(1,6):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    wcss.append(km.inertia_)


# In[163]:


#PLotting WCSS with out sklearn code
import matplotlib.pyplot as plt
plt.plot(range(1,6), wcss)


# In[169]:


X = df.values
wcss=[]
for k in range(1,5):
    random_index = random.sample(range(0, df.shape[0]), k)
    initial_cent = X[random_index]

    for m in range(100):
        
        # Assign Cluster
        assign_cluster =[]
        for row in X:
            distance=[]
            for centroid in initial_cent:
                distance.append(np.sqrt(np.dot(row-centroid, row-centroid)))
            index = distance.index(min(distance))
            assign_cluster.append(index)
        assign_cluster = np.array(assign_cluster)           
        
        # New centroids
        old_cent = initial_cent
        uni_cluster_no = np.unique(assign_cluster)
        cen_new =[]
        for cluster_no in uni_cluster_no:
            cen_new.append(X[assign_cluster == cluster_no].mean(axis=0))

        initial_cent = np.array(cen_new)
        
        # Check convergence
        if(old_cent == initial_cent).all():
            wcss1 = 0
            for i in uni_cluster_no:
                X1=X[assign_cluster==i]
                for j in X1:
                    wcss1 = wcss1 + np.dot((j - initial_cent[i]),(j - initial_cent[i]))
            wcss.append(wcss1)
            break

    print('Total iteration run:',m+1)
    print('Final Centroids are:',cen_new)
print(wcss)


# In[170]:


wcss


# In[172]:


#PLotting WCSS with out own written code 
plt.plot(range(1,5), wcss)


# In[ ]:




