import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn import linear_model

#Loading diabetes dataset
diabetes = load_diabetes()

#Data split into testing and training data.
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_Y_train = diabetes.target[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_Y_test = diabetes.target[-20:]

#Trying to write a replacement function for linear.model.LinearRegression()
average = (np.mean(diabetes_X_train) * np.mean(diabetes_Y_train) - np.mean(diabetes_X_train * diabetes_Y_train))/((np.mean(diabetes_X_train))**2 - np.mean(diabetes_X_train**2))
b = np.mean(diabetes_Y_train) - average * np.mean(diabetes_X_train)

#train linear regression object
average.fit(diabetes_X_train, diabetes_Y_train)

#average square error
mse = np.mean((average.predict(diabetes_X_test) - diabetes_Y_test) ** 2)

#calculate regression score
regr_score = b.score(diabetes_X_test, diabetes_Y_test)

#print statements
print('Coefficients: \n', average.coef_)
print("Mean sqaured error: %.2f", mse)
print("Variance score: %.2f", regr_score)


#graph scatter plots of test data
plt.scatter(diabetes_X_test, diabetes_Y_test, color='green', label="Test")
plt.scatter(diabetes_X_test, diabetes_Y_test, color='red', label='Training')
plt.plot(diabetes_X_test, average.predict(diabetes_X_test), color='blue', linewidth=3, label='Best fit line')
plt.xticks(())
plt.yticks(())
plt.legend()

plt.show()