import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

def linearRegression():
    df = pd.read_csv("FuelConsumption.csv")

    # take a look at the dataset
    df.head()

    # summarize the data
    df.describe()

    cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
    cdf.head(9)

    viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
    viz.hist()
    plt.show()

    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
    plt.xlabel("FUELCONSUMPTION_COMB")
    plt.ylabel("Emission")
    plt.show()

    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()


    #split data set on 80% / 20%
    msk = np.random.rand(len(df)) < 0.8
    print(msk)
    train = cdf[msk]
    test = cdf[~msk]

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit (train_x, train_y)
    # The coefficients
    print ('Coefficients: ', regr.coef_)
    print ('Intercept: ', regr.intercept_)


    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
    plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")

    from sklearn.metrics import r2_score

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_hat = regr.predict(test_x)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )


def multiLinearRegression():
    df = pd.read_csv("FuelConsumption.csv")

    # take a look at the dataset
    df.head()

    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
              'CO2EMISSIONS']]
    cdf.head(9)

    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    msk = np.random.rand(len(df)) < 0.8     #nakladivaem masku
    train = cdf[msk]
    test = cdf[~msk]

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    from sklearn import linear_model
    regr = linear_model.LinearRegression()
    x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(x, y)
    # The coefficients
    print('Coefficients: ', regr.coef_)

    y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"
          % np.mean((y_hat - y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))



    '''
    Try to use a multiple linear regression with the same dataset but this time use __FUEL CONSUMPTION in CITY__ and 
__FUEL CONSUMPTION in HWY__ instead of FUELCONSUMPTION_COMB. Does it result in better accuracy?
    '''

    regr = linear_model.LinearRegression()
    x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
    y = np.asanyarray(train[['CO2EMISSIONS']])

    regr.fit(x, y)
    # The coefficients
    print('Coefficients: ', regr.coef_)

    y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
    x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares: %.2f"
          % np.mean((y_hat - y) ** 2))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(x, y))


def polynomialRegression():
    df = pd.read_csv("FuelConsumption.csv")

    # take a look at the dataset
    df.head()
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    cdf.head(9)

    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    #plt.show()

    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])

    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])

    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)

    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(train_x_poly, train_y)
    # The coefficients
    print('Coefficients: ', clf.coef_)
    print('Intercept: ', clf.intercept_)

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    XX = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX, 2)
    plt.plot(XX, yy, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")

    from sklearn.metrics import r2_score

    test_x_poly = poly.fit_transform(test_x)
    test_y_ = clf.predict(test_x_poly)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_, test_y))



def nonLinearRegression():
    import numpy as np
    x = np.arange(-5.0, 5.0, 0.1)


    ##You can adjust the slope and intercept to verify the changes in the graph
    y = 2 * (x) + 3
    y_noise = 2 * np.random.normal(size=x.size)
    ydata = y + y_noise
    # plt.figure(figsize=(8,6))
    plt.plot(x, ydata, 'bo')
    plt.plot(x, y, 'r')
    plt.ylabel('Dependent Variable')
    plt.xlabel('Indepdendent Variable')
    plt.show()

    x = np.arange(-5.0, 5.0, 0.1)

    ##You can adjust the slope and intercept to verify the changes in the graph
    y = 1 * (x ** 3) + 1 * (x ** 2) + 1 * x + 3
    y_noise = 20 * np.random.normal(size=x.size)
    ydata = y + y_noise
    plt.plot(x, ydata, 'bo')
    plt.plot(x, y, 'r')
    plt.ylabel('Dependent Variable')
    plt.xlabel('Indepdendent Variable')
    plt.show()

    import numpy as np
    import pandas as pd

    df = pd.read_csv("china_gdp.csv")
    df.head(10)

    plt.figure(figsize=(8, 5))
    x_data, y_data = (df["Year"].values, df["Value"].values)
    plt.plot(x_data, y_data, 'ro')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

    def sigmoid(x, Beta_1, Beta_2):
        y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
        return y

    beta_1 = 0.10
    beta_2 = 1990.0

    # logistic function
    Y_pred = sigmoid(x_data, beta_1, beta_2)

    # plot initial prediction against datapoints
    plt.plot(x_data, Y_pred * 15000000000000.)
    plt.plot(x_data, y_data, 'ro')

    # Lets normalize our data
    xdata = x_data / max(x_data)
    ydata = y_data / max(y_data)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    # print the final parameters
    print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

    x = np.linspace(1960, 2015, 55)
    x = x / max(x)
    plt.figure(figsize=(8, 5))
    y = sigmoid(x, *popt)
    plt.plot(xdata, ydata, 'ro', label='data')
    plt.plot(x, y, linewidth=3.0, label='fit')
    plt.legend(loc='best')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()

    # Can you calculate what is the accuracy of our model?

    # write your c# split data into train/test
    msk = np.random.rand(len(df)) < 0.8
    train_x = xdata[msk]
    test_x = xdata[~msk]
    train_y = ydata[msk]
    test_y = ydata[~msk]

    # build the model using train set
    popt, pcov = curve_fit(sigmoid, train_x, train_y)

    # predict using test set
    y_hat = sigmoid(test_x, *popt)

    # evaluation
    print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
    from sklearn.metrics import r2_score
    print("R2-score: %.2f" % r2_score(y_hat, test_y))


if __name__ == '__main__':
    #linearRegression()
    #multiLinearRegression()
    #polynomialRegression
    nonLinearRegression()
