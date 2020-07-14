
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\Deepak\.spyder-py3\advertising.csv")
data.head()

data["Newspaper"].plot.hist()
data.info()
data["Newspaper"].fillna(data["Newspaper"].median(), inplace=True)
data.info()

data["TV"].plot.hist()
data["Sales"].plot.hist()
data["Radio"].plot.hist()

plt.scatter(data["TV"], data["Sales"])
plt.title("Tv V/s Sales")
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(data["Radio"], data["Sales"])
plt.title("Radio V/s Sales")
plt.show()

Y= data["Sales"]
X = data.drop(["Sales"], axis =1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=.30, random_state=42)
print("shape of train: ", x_train.shape)
print("shape of test: ", x_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
result_data = pd.DataFrame({"Actual Sales":y_test,"predicted sales":y_pred})
print(result_data.head())
print(result_data.tail())

# getting vif-factor of x-parameter
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["VIF Factor"]=[variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif.round(1))

import statsmodels.api as sm
X_final=sm.add_constant(x_train)
lr= sm.OLS(y_train,X_final).fit()
print("summary")
print(lr.summary())

# taking x-papameter as tv,radio leave(Newspaper)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
Y=data["Sales"]
X=data.loc[0:,["TV","Radio"]]
X.columns

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=.30, random_state=42)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print(y_pred)

X.head()
Y.head()

tv=float(input("Enter amount : "))
radio=float(input("Enter amount : "))
test=[tv,radio]
prediction=model.predict([test,])
print("prediction value is:",prediction)

model.score(x_train,y_train)
residual = y_test-y_pred
print(residual)
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.metrics import r2_score
print("R-square value is :",r2_score(y_test,y_pred))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\Deepak\.spyder-py3\advertising.csv")
data.head()

X=data.loc[0:,["TV"]].values
Y=data.loc[0:,["Sales"]].values
print(type(X))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=.30, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import r2_score
print("R-square value is :",r2_score(y_test,y_pred))

plt.figure(figsize=(10,10))
plt.scatter(y_test,x_test,color="blue",marker='o')
plt.title("Linear regression analysis")
plt.xlabel("X-test data")
plt.ylabel("Y-test/y-pred")
plt.show()

from sklearn.datasets import load_boston
data=load_boston()
print(data)
