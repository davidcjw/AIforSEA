import pandas as pd
import geohash as gh
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import statistics as ss
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# plt.style.use('fivethirtyeight')
# %matplotlib inline

df = pd.read_csv('training.csv')

def get_hour(timestamp):
    if ":" not in timestamp[:2]:
        return timestamp[:2]
    else:
        return timestamp[:1]

def get_minute(timestamp):
    if ":" not in timestamp[-2:]:
        return timestamp[-2:]
    else:
        return timestamp[-1]

def day_of_week(day):
    options = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    return options[day%7]

df['lat_long'] = df['geohash6'].apply(gh.decode_exactly)
df['x_coord'] = df['lat_long'].apply(lambda x: x[0])
df['y_coord'] = df['lat_long'].apply(lambda x: x[1])
df['hour'] = df['timestamp'].apply(get_hour)
df['minute'] = df['timestamp'].apply(get_minute)
convert_dtypes = {'x_coord': float, 'y_coord': float, 'day': int, 'hour': int, 'minute': int}
df = df.astype(convert_dtypes)
df['hour_min'] = df['hour']*4 + df['minute']/15
df = df.astype({'hour_min': int})
df['day_hour_min'] = (df['day']-1)*96 + df['hour_min']
df = df.astype({'day_hour_min': int})
df['day_of_week'] = df['day'].apply(day_of_week)
df.sort_values(by=['day', 'hour', 'minute'], inplace=True)

scalar = MinMaxScaler()
x_tfm = scalar.fit_transform(np.array(df['x_coord']).reshape(-1,1))
y_tfm = scalar.fit_transform(np.array(df['y_coord']).reshape(-1,1))
df['x_tfm'] = x_tfm
df['y_tfm'] = y_tfm
time_tfm = scalar.fit_transform(np.array(df['hour_min']).reshape(-1,1))
df['time_tfm'] = time_tfm

X = df[['x_tfm','y_tfm','time_tfm', 'day_of_week']]
y = df['demand']
labelencoder = LabelEncoder()
X.iloc[:, 3] = labelencoder.fit_transform(X.iloc[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
X = pd.DataFrame(X)
split_ratio = 0.8
X_train = X.iloc[:int(round(split_ratio*len(df),0))]
X_test = X.iloc[int(round(split_ratio*len(df),0)):]
y_train = y.iloc[:int(round(split_ratio*len(df),0))]
y_test = y.iloc[int(round(split_ratio*len(df),0)):]
y_train = y_train.reset_index()
y_train = y_train['demand']
y_test = y_test.reset_index()
y_test = y_test['demand']

forest = RandomForestRegressor(n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_features='auto',
             max_depth=100, bootstrap=True)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
print(y_pred_forest)
# print('Random Forest RMSE: (50 trees)', np.sqrt(mean_squared_error(y_test, y_pred_forest)))