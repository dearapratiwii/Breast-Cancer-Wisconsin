#!/usr/bin/env python
# coding: utf-8

# ### *Preprocessing, Visualisasi,* dan *Feature Extraction*
# Data *Breast Cancer Wisconsin Diagnostic Dataset Kaggle*
# 
# **Oleh:** 
# 
# Dea Restika Augustina Pratiwi (06211740000023)
# 
# 
# ##### Download Data : [kaggle](https://drive.google.com/file/d/1sqrmytAewju0K6fnUoiQhp_GK2Dy__nz/view)

# **Impor Library**

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# **Impor Data**

# In[2]:


data = pd.read_excel('D:/breastcancer.xlsx')
data


# In[3]:


data.info()


# ### <center>A. *PREPROCESSING*</center>

# ### 1. Deteksi *Missing Value*

# In[4]:


np.sum(data.isnull())


# Dari output di atas dapat diketahui bahwa terdapat missing value pada variabel unnamed: 32. Namun, yang terjadi adalah keseluruhan pengamatan pada variabel tersebut terjadi missing value, hal ini dapat diamati dari jumlah missing value = jumlah entri data. Maka karena tidak ada informasi sama sekali mengenai variabel ini, variabel akan dihapus dari pegamatan. Selain itu akan dilakukan penghapusan pada vriabel ID karena tidak digunakan dalam analisis.

# In[5]:


data.drop("Unnamed: 32", axis = 1, inplace = True)
data.drop("id", axis = 1, inplace = True)


# In[6]:


np.sum(data.isnull())


# Sekarang, dalam data sudah tidak ada lagi missing value

# ### 2. Deteksi *Outlier*
# Deteksi *outlier* akan dilakukan dengan menggunakan *boxplot*
# Note : Karena terdapat 10 variabel yang akan dideteksi outliernya dan tidak memungkinkan jika dijadikan 1 plot, maka digunakan subplot

# In[7]:


plt.figure(figsize = (6,11))
plt.subplot(521)
sns.boxplot(x = data['radius'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplots_adjust(wspace = 0.6)
plt.subplot(522)
sns.boxplot(x = data['texture'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(523)
sns.boxplot(x = data['perimeter'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(524)
sns.boxplot(x = data['area'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(525)
sns.boxplot(x = data['smoothness'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(526)
sns.boxplot(x = data['compactness'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(527)
sns.boxplot(x = data['concavity'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(528)
sns.boxplot(x = data['concave points'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(529)
sns.boxplot(x = data['symmetry'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.subplot(5, 2, 10)
sns.boxplot(x = data['fractal_dimension'], saturation = 1, width = 0.8, color = 'c', orient = 'v')
plt.show()


# In[8]:


## Alternatif lain dari boxplot
data = pd.read_excel('D:/breastcancer.xlsx')
data
y = data.diagnosis 
data_dia = y
list = ['Unnamed: 32','id','diagnosis']
x = data.drop(list,axis = 1 )
data = x
data_n_2 = (data - data.mean()) / (data.std())
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,6))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# ### <center>B. *Summary Statistics*</center>

# In[9]:


data = pd.read_excel('D:/breastcancer.xlsx')
list = ['Unnamed: 32','id','diagnosis']
data = data.drop(list,axis = 1 )
data.describe()


# ### <center>C. Visualisasi Data</center>

# In[16]:


data = pd.read_excel('D:/breastcancer.xlsx')
list = ['Unnamed: 32','id']
data = data.drop(list,axis = 1 )
data.diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = True)
palette ={0 : 'lightblue', 1 : 'gold'}
edgecolor = 'grey'

fig = plt.figure(figsize=(8,8))

plt.subplot(221)
ax1 = sns.scatterplot(x = data['perimeter'], y = data['fractal_dimension'], hue = "diagnosis",
                    data = data, palette = palette, edgecolor=edgecolor)
plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
plt.title('perimeter vs radius')
plt.subplot(222)
ax2 = sns.scatterplot(x = data['texture'], y = data['area'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs radius')
plt.subplot(223)
ax3 = sns.scatterplot(x = data['smoothness'], y = data['compactness'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('texture mean vs texture')
plt.subplot(224)
ax4 = sns.scatterplot(x = data['concavity'], y = data['symmetry'], hue = "diagnosis",
                    data = data, palette =palette, edgecolor=edgecolor)
plt.title('area mean vs radius')

plt.savefig('1')
plt.show()


# In[27]:


data = pd.read_excel('D:/breastcancer.xlsx')
data.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
m = plt.hist(data[data["diagnosis"] == "M"].perimeter,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].perimeter,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Perimeter")
plt.ylabel("Frequency")
plt.title("Histogram dari Perimeter")
plt.show()
frequent_malignant_perimeter = m[0].max()
index_frequent_malignant_perimeter = list(m[0]).index(frequent_malignant_perimeter)
most_frequent_malignant_perimeter = m[1][index_frequent_malignant_perimeter]
print("Most frequent malignant radius mean is: ",most_frequent_malignant_perimeter)


# ### <center>D. Analisis *Cluster*</center>

# In[17]:


data = pd.read_excel('D:/breastcancer.xlsx')
data.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
dataWithoutLabels = data.drop(["diagnosis"], axis = 1)
wcss = []

for k in range(1, 15):
    kmeansForLoop = KMeans(n_clusters = k)
    kmeansForLoop.fit(dataWithoutLabels)
    wcss.append(kmeansForLoop.inertia_)

plt.figure(figsize = (5, 5))
plt.plot(range(1, 15), wcss)
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.show()


# In[4]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing
from subprocess import check_output
data = pd.read_excel('D:/breastcancer.xlsx')
data = data.drop('id',axis=1)
data = data.drop('Unnamed: 32',axis=1)
datas = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))
datas.columns = list(data.iloc[:,1:32].columns)
datas['diagnosis'] = data['diagnosis']
data_drop = datas.drop('diagnosis',axis=1)
X = data_drop.values
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1, perplexity=40, n_iter= 4000)
Y = tsne.fit_transform(X)


# In[5]:


from sklearn.cluster import KMeans
kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kY = kmns.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(Y[:,0],Y[:,1],  c=kY, cmap = "jet", edgecolor = "None", alpha=0.35)
ax1.set_title('k-means clustering plot')

ax2.scatter(Y[:,0],Y[:,1],  c = datas['diagnosis'], cmap = "jet", edgecolor = "None", alpha=0.35)
ax2.set_title('Actual clusters')


# In[13]:


dataWithoutLabels = data.loc[:,['area','texture']]
kmeans = KMeans(n_clusters = 2)
clusters = kmeans.fit_predict(dataWithoutLabels)
dataWithoutLabels["type"] = clusters
dataWithoutLabels["type"].unique()


# In[14]:


plt.figure(figsize = (15, 10))
plt.scatter(dataWithoutLabels["area"][dataWithoutLabels["type"] == 0], dataWithoutLabels["texture"][dataWithoutLabels["type"] == 0], color = "red")
plt.scatter(dataWithoutLabels["area"][dataWithoutLabels["type"] == 1], dataWithoutLabels["texture"][dataWithoutLabels["type"] == 1], color = "green")
plt.xlabel('area')
plt.ylabel('texture')
plt.show()


# In[17]:


plt.figure(figsize = (15, 10))
plt.scatter(dataWithoutLabels["area"], dataWithoutLabels["texture"], c = clusters, alpha = 0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = "red", alpha = 1)
plt.xlabel('area')
plt.ylabel('texture')
plt.show()


# In[18]:


dataWithoutTypes = dataWithoutLabels.drop(["type"], axis = 1)
dataWithoutTypes.head()


# In[19]:


from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(dataWithoutTypes, method = "ward")
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# In[59]:


corr_matrix=data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Korelasi Antar Variabel")
plt.show()


# ### <center>E. *Feature Extraction dengan PCA*</center>

# #### Analisis PCA

# In[32]:


data = pd.read_excel('D:/breastcancer.xlsx')
data.drop(["Unnamed: 32", "id", "diagnosis"], axis = 1, inplace = True)
f,ax = plt.subplots(figsize=(7, 7))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[34]:


data = pd.read_excel('D:/breastcancer.xlsx')
data.drop(["Unnamed: 32", "id", "diagnosis"], axis = 1, inplace = True)
X = StandardScaler().fit_transform(data)
 
pca = PCA(n_components=0.85, whiten=True) 
 
X_pca = pca.fit_transform(X)

print('Original number of features:', X.shape[1]) 
print('Reduced number of features:', X_pca.shape[1])


# In[37]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("D:/breastcancer.xlsx", index_col = 'id')
df.drop('Unnamed: 32',axis = 1 ,inplace = True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B':0})
X = df.drop('diagnosis',axis = 1)
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

pca = decomposition.PCA(n_components=2)
X_pca_scaled = pca.fit_transform(X_scaled)

print('Projecting %d-dimensional data to 2D' % X_scaled.shape[1])

plt.figure(figsize=(12,10))
plt.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], c=df['diagnosis'], alpha=0.7, s=40);
plt.colorbar()
plt.title('MNIST. PCA projection');
pca = decomposition.PCA().fit(X_scaled)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 29)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(6, c='b')
plt.axhline(0.8, c='r')
plt.show();


# In[23]:


data = pd.read_excel('D:/breastcancer.xlsx')
data.drop(["Unnamed: 32", "id", "diagnosis"], axis = 1, inplace = True)
X = StandardScaler().fit_transform(data)
 
pca = PCA(n_components=0.80, whiten=True) 
 
X_pca = pca.fit_transform(X)

print('Original number of features:', X.shape[1]) 
print('Reduced number of features:', X_pca.shape[1])

