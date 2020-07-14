
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r".spyder-py3\Mall_Customers.csv")
print(data)

X=data.iloc[0:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,10):
    model=KMeans(n_clusters=i, init='k-means++', random_state=42)
    model.fit(X)
    wcss.append(model.inertia_)
    
plt.plot(range(1,10), wcss)
plt.title("ELBOW METHOD")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS value")
plt.show()

model=KMeans(n_clusters=5, init='k-means++', random_state=42)
pred_cluster = model.fit_predict(X)
print(pred_cluster)

d=pd.DataFrame({"cluster":pred_cluster})
data=pd.concat((data,d),axis=1)

plt.scatter(X[pred_cluster==0,0],X[pred_cluster==0,1],s=100,c='red',label='cluster0')

plt.scatter(X[pred_cluster==1,0],X[pred_cluster==1,1],s=100,c='blue',label='cluster1')

plt.scatter(X[pred_cluster==2,0],X[pred_cluster==2,1],s=100,c='yellow',label='cluster2')

plt.scatter(X[pred_cluster==3,0],X[pred_cluster==3,1],s=100,c='magenta',label='cluster3')

plt.scatter(X[pred_cluster==4,0],X[pred_cluster==4,1],s=100,c='black',label='cluster4')

plt.title("Cluster of customer")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()

####################################################################
# customer segment on the basis of gender and spending score

data=pd.read_csv(r".spyder-py3\Mall_Customers.csv")
print(data)

data["Gender"]=data.Gender.map({"Male":1,"Female":0})
X=data.iloc[0:,[1,4]].values

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,10):
    model=KMeans(n_clusters=i, init='k-means++', random_state=42)
    model.fit(X)
    wcss.append(model.inertia_)
    
plt.plot(range(1,10), wcss)
plt.title("ELBOW METHOD")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS value")
plt.show()

model=KMeans(n_clusters=3, init='k-means++', random_state=42)
pred_cluster = model.fit_predict(X)
print(pred_cluster)

d=pd.DataFrame({"cluster":pred_cluster})
data=pd.concat((data,d),axis=1)

plt.scatter(X[pred_cluster==0,0],X[pred_cluster==0,1],s=100,c='red',label='cluster0')

plt.scatter(X[pred_cluster==1,0],X[pred_cluster==1,1],s=100,c='blue',label='cluster1')

plt.scatter(X[pred_cluster==2,0],X[pred_cluster==2,1],s=100,c='yellow',label='cluster2')


plt.title("Cluster of customer")
plt.xlabel("Gender")
plt.ylabel("spending score")
plt.legend()
plt.show()

########################################################################
#  customer segment on the basis of age and spending score

data=pd.read_csv(r".spyder-py3\Mall_Customers.csv")
print(data)

X=data.iloc[0:,[2,4]].values

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,10):
    model=KMeans(n_clusters=i, init='k-means++', random_state=42)
    model.fit(X)
    wcss.append(model.inertia_)
    
plt.plot(range(1,10), wcss)
plt.title("ELBOW METHOD")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS value")
plt.show()

model=KMeans(n_clusters=4, init='k-means++', random_state=42)
pred_cluster = model.fit_predict(X)
print(pred_cluster)

d=pd.DataFrame({"cluster":pred_cluster})
data=pd.concat((data,d),axis=1)

plt.scatter(X[pred_cluster==0,0],X[pred_cluster==0,1],s=100,c='red',label='cluster0')

plt.scatter(X[pred_cluster==1,0],X[pred_cluster==1,1],s=100,c='blue',label='cluster1')

plt.scatter(X[pred_cluster==2,0],X[pred_cluster==2,1],s=100,c='yellow',label='cluster2')

plt.scatter(X[pred_cluster==3,0],X[pred_cluster==3,1],s=100,c='magenta',label='cluster3')

plt.title("Cluster of customer")
plt.xlabel("age")
plt.ylabel("spending score")
plt.legend()
plt.show()

######################################################################
# dendogram curve

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r".spyder-py3\Mall_Customers.csv")

x=data.iloc[0:,[3,4]].values

import scipy.cluster.hierarchy as sch

dendo = sch.dendrogram(sch.linkage(x,method='ward'))

plt.title("Dendogram curve")
plt.xlabel("datapoints")
plt.ylabel("Euclidean Distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

y_pred=model.fit_predict(x)
print(y_pred)

