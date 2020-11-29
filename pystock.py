import  numpy   as  np
import  pandas  as  pd 
from    sklearn.tree    import  DecisionTreeRegressor
from    sklearn.linear_model    import  LinearRegression
from    sklearn.model_selection import  train_test_split
import  matplotlib.pyplot   as plt
import  yfinance    as yf
import  yahoofinancials


stock_df = yf.download('NFLX', start='2019-01-01', end=(pd.to_datetime('today')), progress=False, auto_adjust = True)
forecast_out = 30
stock_df['Prediction'] = stock_df[['Close']].shift(-forecast_out)

X = np.array(stock_df.drop(['Prediction'], 1))[:-forecast_out]
y = np.array(stock_df['Prediction'])[:-forecast_out]
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=1234)
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)
X_future = stock_df.drop(['Prediction'], 1)[:-forecast_out]
X_future = X_future.tail(forecast_out)
X_future = np.array(X_future)
tree_prediction = tree.predict(X_future)
print(tree_prediction)
print()
lr_prediction = lr.predict(X_future)
print(lr_prediction)
lr_confidence = lr.score(x_test, y_test)
print("Prediction accuracy ", lr_confidence)
