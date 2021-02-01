# IMPORT LIBRARIES:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


# IMPORT HEIGHT AND WEIGHT DATASET:
data = pd.read_csv('Height_Age_Dataset.csv')
data.head()

# STORE THE DATA IN THE FORM OF DEPENDENT AND INDEPENDENT VARIABLES SEPARATELY:
Xaxis = data.iloc[:, 0:1].values
yaxis = data.iloc[:, 1].values

# SPLIT DATASET INTO TRAINING AND TEST DATASET:
X_train, X_test, y_train, y_test = train_test_split(Xaxis, yaxis, test_size=0.3, random_state=0)

# FIT THE SIMPLE LINEAR REGRESSION MODEL:
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

# ADD POLYNOMIAL TERM TO EQUATION/MODEL:
polynom = PolynomialFeatures(degree=2)
X_polynom = polynom.fit_transform(X_train)

X_polynom

# FIT THE POLYNOMIAL REGRESSION MODEL:
PolyReg = LinearRegression()
PolyReg.fit(X_polynom, y_train)

# VISUALIZE THE POLYNOMIAL REGRESSION RESULT:
plt.scatter(X_train, y_train, color='green')

plt.plot(X_train, PolyReg.predict(polynom.fit_transform(X_train)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Age')
plt.ylabel('Height')

plt.show()

# PREDICT HEIGHT FROM TEST DATASET W.R.T SIMPLE LINEAR REGRESSION:
y_predict_slr = LinReg.predict(X_test)

# MODEL EVALUATION USING R-SQAURE FOR SIMPLE LINEAR REGRESSION:
r_sqaured = metrics.r2_score(y_test, y_predict_slr)
print('R-Sqaure Error associated with Polynomial Regression is:', r_sqaured)

# PREDICT HEIGHT BASED ON AGE USING LINEAR REGRESSION:
LinReg.predict([[38]])

# PREDICTING HEIGHT BASED ON AGE USING POLYNOMIAL REGRESSION:
PolyReg.predict(polynom.fit_transform([[38]]))
