import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
df_1 = pd.read_csv("PAR_NYC.csv")
df_2 = pd.read_csv("PAR_SVO.csv")
df_3 = pd.read_csv("SVO_NYC.csv")
df_4 = pd.read_csv("SVO_RUH.csv")
df_5 = pd.read_csv("NYC_PAR.csv")
df_6 = pd.read_csv("NYC_SVO.csv")
df_7 = pd.read_csv("RUH_NYC.csv")
df_8 = pd.read_csv("RUH_PAR.csv")
df_9 = pd.read_csv("RUH_SVO.csv")
df_10 = pd.read_csv("SVO_PAR.csv")
df_11 = pd.read_csv("PAR_RUH.csv")
df_12 = pd.read_csv("NYC_RUH.csv")

print(f"{df_1['Source'][0]} => {df_1['Destination'][0]} route has {df_1.shape[0]} trips")
print(f"{df_2['Source'][0]} => {df_2['Destination'][0]} route has {df_2.shape[0]} trips")
print(f"{df_3['Source'][0]} => {df_3['Destination'][0]} route has {df_3.shape[0]} trips")
print(f"{df_4['Source'][0]} => {df_4['Destination'][0]} route has {df_4.shape[0]} trips")
print(f"{df_5['Source'][0]} => {df_5['Destination'][0]} route has {df_5.shape[0]} trips")
print(f"{df_6['Source'][0]} => {df_6['Destination'][0]} route has {df_6.shape[0]} trips")
print(f"{df_7['Source'][0]} => {df_7['Destination'][0]} route has {df_7.shape[0]} trips")
print(f"{df_8['Source'][0]} => {df_8['Destination'][0]} route has {df_8.shape[0]} trips")
print(f"{df_9['Source'][0]} => {df_9['Destination'][0]} route has {df_9.shape[0]} trips")
print(f"{df_10['Source'][0]} => {df_10['Destination'][0]} route has {df_10.shape[0]} trips")
print(f"{df_11['Source'][0]} => {df_11['Destination'][0]} route has {df_11.shape[0]} trips")
print(f"{df_12['Source'][0]} => {df_12['Destination'][0]} route has {df_12.shape[0]} trips")


def clean_duration(duration):
    duration = list(duration)
    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

    d = []
    for i in range(len(duration)):
        d.append(duration_hours[i] * 60 + duration_mins[i])

    return d


# convert price to numerical format in USD
def clean_price(price):
    price = price.str.replace(',', '', regex=True)
    price = price.str.replace('SAR', '', regex=True)
    price = price.str.strip()
    price = round(pd.to_numeric(price) / 3.75, 2)
    return price


# convert date to datetime format
def clean_date(date):
    date = pd.to_datetime(date)
    return date


# get price quantile to deal with outliers
def get_price_quantile(price):
    Q1 = price.quantile(0.25)
    Q3 = price.quantile(0.75)
    IQR = Q3 - Q1
    lower_lim = Q1 - 1.5 * IQR
    upper_lim = Q3 + 1.5 * IQR
    return (lower_lim, upper_lim)


# get average of each airline
def get_avg_per_airline(x):
    # average for trips with multiple airlines
    multiple_airlines = x[x["Airline"].str.contains(",")]
    b = list(multiple_airlines["Airline"].str.split(","))
    d = []  # Airline 1
    e = []  # Airline 2
    for i in range(len(b)):
        d.append(b[i][0])
        e.append(b[i][1])
    for i in range(len(e)):
        e[i] = e[i].strip()
    m_airlines = list(set(d)) + list(set(e))
    column_names = ["Airline", "Average Price"]
    t_ = pd.DataFrame(columns=column_names)
    for airline in m_airlines:
        t = pd.DataFrame(x[x["Airline"].str.contains(airline)]["Airline"])
        t["Average Price"] = x[x["Airline"].str.contains(airline)]["Price"].mean()
        t_ = t_.append(t)
    t__ = t_.groupby("Airline", as_index=False)["Average Price"].mean()
    k = multiple_airlines.copy()
    k = k.merge(t__, on="Airline", how="left")

    # average for trips with single airlines
    single_airlines = x[~x["Airline"].str.contains(",")]
    avg_per_airline = single_airlines.groupby("Airline", as_index=False)["Price"].mean()
    avg_per_airline = avg_per_airline.rename(columns={"Price": "Average Price"})
    temp = single_airlines.copy()
    temp = temp.merge(avg_per_airline, on='Airline', how="left")

    temp_1 = temp.groupby("Airline", as_index=False)["Average Price"].mean()
    k_1 = k.groupby("Airline", as_index=False)["Average Price"].mean()
    k_temp = pd.concat([k_1, temp_1])
    y = x.merge(k_temp, on="Airline")

    return y


dfs_raw = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12]
dfs = []
for df in dfs_raw:
    df.drop_duplicates()  # drop duplicate rows
    df["Duration"] = clean_duration(df["Duration"])  # convert duration to numerical minutes format
    df["Price"] = clean_price(df["Price"])  # convert price to numerical format in USD
    df["Date"] = clean_date(df["Date"])  # convert date to datetime format
    dfs.append(get_avg_per_airline(df))  # get average per airline

k = 0
figure, axis = plt.subplots(4, 3, figsize=(15, 15))
for i in range(4):
    for j in range(3):
        axis[i, j].boxplot(dfs[k]['Price'])
        axis[i, j].set_title(f"{dfs[k]['Source'][0]} TO {dfs[k]['Destination'][0]}")
        k += 1

# plt.show()
lower = []
upper = []
for df in dfs:
    x = get_price_quantile(df['Price'])
    lower.append(x[0])
    upper.append(x[1])

k = 0
for df in dfs:
    low = df['Price'] < lower[k]
    up = df['Price'] > upper[k]
    df['Price'] = df['Price'][~(low | up)]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    k += 1

k = 0
figure, axis = plt.subplots(4, 3, figsize=(15, 15))
for i in range(4):
    for j in range(3):
        axis[i, j].boxplot(dfs[k]['Price'])
        axis[i, j].set_title(f"{dfs[k]['Source'][0]} TO {dfs[k]['Destination'][0]}")
        k += 1

plt.show()
df = pd.concat(dfs)
df_c = pd.concat(dfs)
#sns.jointplot(df_c,x=df["Total stops"],y=df["Source"])
#plt.show()
print(df.isnull().sum())
print(df["Source"].value_counts())
print(df["Destination"].value_counts())
print(df["Total stops"].value_counts())
print(df["Total stops"].unique())

source = df[["Source"]]
source = pd.get_dummies(source, drop_first=True)
df['Source Name'] = pd.factorize(df['Source'])[0]
print(source.head())
destination = df[["Destination"]]
destination = pd.get_dummies(destination, drop_first=True)
print(destination.head(10))
df.replace({"nonstop ": 0, "1 stop ": 1, "2 stops ": 2, "3 stops ": 3}, inplace=True)
final_df = pd.concat([df, source, destination], axis=1).reset_index(drop=True)
print(final_df)
final_df.drop(["Source", "Destination", "Date"], axis=1, inplace=True)
print(final_df)
print(final_df.shape)
print(final_df.isnull().sum())
print(final_df.columns)

# fig5, axs = plt.subplots(3, 3, figsize=(15, 9.5))
# plt1 = sns.scatterplot(x=final_df.Duration, y=final_df.Price,data=final_df, ax=axs[0, 0])
# plt2 = sns.scatterplot(data=final_df, x=final_df['Total stops'], y=final_df.Price, ax=axs[0, 1])
# plt3 = sns.scatterplot(data=final_df, x=final_df.Source_PAR, y=final_df.Price, ax=axs[0, 2])
# plt4 = sns.scatterplot(data=final_df, x=final_df['Average Price'], y=final_df.Price, ax=axs[1, 0])
# plt5 = sns.scatterplot(data=final_df, x=final_df.Source_RUH, y=final_df.Price, ax=axs[1, 1])
# plt6 = sns.scatterplot(data=final_df, x=final_df.Source_SVO, y=final_df.Price, ax=axs[1, 2])
# plt7 = sns.scatterplot(data=final_df, x=final_df.Destination_PAR, y=final_df.Price, ax=axs[2, 0])
# plt8 = sns.scatterplot(data=final_df, x=final_df.Destination_RUH, y=final_df.Price, ax=axs[2, 1])
# plt9 = sns.scatterplot(data=final_df, x=final_df.Destination_SVO, y=final_df.Price, ax=axs[2, 2])
# plt.show()


X = final_df[['Duration', 'Total stops', 'Source_PAR', 'Average Price',
              'Source_RUH', 'Source_SVO', 'Destination_PAR', 'Destination_RUH',
              'Destination_SVO']]
y = final_df["Price"]
# plt.figure(figsize=(18, 18))
#
# sns.heatmap(final_df.corr(), annot=True, cmap="coolwarm")
#
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# rt = RandomForestRegressor(n_estimators=20,random_state=1)
#rt = RandomForestRegressor()
# rt = RandomForestRegressor(n_estimators=20,random_state=1)

# rt = DecisionTreeRegressor(random_state=1,max_depth=23)
#rt = DecisionTreeRegressor()
# rt = ExtraTreesRegressor(n_estimators=20,random_state=1)
rt = ExtraTreesRegressor()
# rt = KNeighborsRegressor(n_neighbors=5)
#rt = KNeighborsRegressor(n_neighbors=3)
# rt = AdaBoostRegressor(n_estimators=25,random_state=1)
#rt = AdaBoostRegressor()
## rt=BaggingRegressor(random_state=1,n_estimators=25,bootstrap=10000,bootstrap_features=10000)
rt = BaggingRegressor()
rt.fit(X_train, y_train)
predict = rt.predict(X_test)
print('Part 2 price----------------')
print(X.head(5))
# print(X.columns)
print('Part 2 -----------------------------------------------')
print('Variance between Test value and predict')
print(metrics.explained_variance_score(y_test, predict) * 100)
print(rt.predict([[770, 1, 1, 441.3, 0, 0, 0, 0, 0]])[0], '$USD')

# sns.pairplot(final_df)
# plt.show()
# With hyperparameter tuning

# crossvlidation = KFold(n_splits=10, shuffle=True, random_state=1)
# for depth in range(1, 25):
#   treeD = tree.DecisionTreeRegressor(max_depth=depth, random_state=1)
#   if treeD.fit(X, y).tree_.max_depth < depth:
#       break
#   score = np.mean(cross_val_score(treeD, X, y, scoring='neg_mean_squared_error', cv=crossvlidation, n_jobs=1))
#   print(depth, '............', score)
#MSE = (mean_squared_error(y_test,predict)*0.006)
##MSE = np.square(np.subtract(y_test,predict)).mean()
#RMSE = np.sqrt(MSE)
## print(y_test.shape)
#MAE = (mean_absolute_error(y_test,predict)*0.006*100)#100/15030
### print('Without hyperparameter tuning')
####print('With hyperparameter tuning')
#print('In percentage of 100 ')
#print('MSE = ', MSE)
#print('.........................')
#print('MAE = ', MAE)
#print('.........................')
#print('RMSE = ', RMSE)
#
# def mse(y_test, predict):
#    y_test = np.array(y_test)
#    predict = np.array(predict)
#    differences = np.subtract(y_test, predict)
#    squared_differences = np.square(differences)
#    return squared_differences.mean()

# print(mse(y_test,predict))


rf = RandomForestRegressor()
rf.fit(X_train,y_train)

print('X Train    vs   Y Train')
print(f'Train score {rf.score(X_train,y_train)*100}')
print('X Test    vs   Y Test')
print(f'Test score {rf.score(X_test, y_test)*100}')
sns.jointplot(data=final_df,x=(rf.score(X_train,y_train)*100),y=(rf.score(X_test, y_test)*100))

plt.show()
df_c['Airline Number'] = pd.factorize(df_c['Airline'])[0]
df_c['Source Number'] = pd.factorize(df_c['Source'])[0]
df_c['Destination Number'] = pd.factorize(df_c['Destination'])[0]
df_c['Stop Numbers'] = pd.factorize(df_c['Total stops'])[0]
df_c['Source Numbers'] = pd.factorize(df_c['Source'])[0]
print('.......................For AirLine')
print(df_c.head(5))
print(df_c.tail(5))
print('............................................')
Xc = df_c[['Duration', 'Airline Number']]
yc = df_c['Airline']
plt.show()
sns.scatterplot(data=df_c,x='Airline',y='Duration')
plt.show()
sns.boxplot(data=df_c,x='Airline',y='Duration')
plt.show()


plt.figure(figsize=(18, 18))
sns.heatmap(df_c.corr(), annot=True, cmap="coolwarm")
plt.show()

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.3, random_state=42)

##dtree = DecisionTreeClassifier(random_state=1,max_depth=23)
#dtree = DecisionTreeClassifier()
dtree = RandomForestClassifier()
##dtree = BaggingClassifier()
#dtree =AdaBoostClassifier()
##dtree =KNeighborsClassifier()
##dtree =ExtraTreesClassifier()
dtree.fit(Xc_train, yc_train)
Predict = dtree.predict(Xc_test)
#
#print('\tExperimential\t\n')
#
#cm = confusion_matrix(yc_test, Predict)
#plt.figure(figsize=(7, 5))
#sns.heatmap(cm, annot=True)
#plt.xlabel('Predicted')
#plt.ylabel('True')

# print('confusion_matrix')
# print('default parameter')
#print(cm)
# print('......................................')
# print('classification_report')
#print(classification_report(yc_test, Predict))
# print('......................................')
print('Enter the Airline number\n')
Number = input('.......')
print('Enter time in minutes')
Duration = input('------')
print(dtree.predict([[Duration, Number]])[0], '--Airline')
air = dtree.predict([[Duration, Number]])[0]
print(df_c[df_c['Airline'].between(air, air)].head(5))
Stops = int(input('[0 to 3]enter Total stops='))
par = int(input('enter Source_PAR[0 or 1]'))
ruh = int(input('enter Source_RUH[0 or 1]'))
sov = int(input('enter Source_SVO[0 or 1]'))
d_par = int(input('enter Destination_PAR[0 or 1]'))
d_ruh = int(input('enter Destination_RUH[0 or 1]'))
d_sov = int(input('enter Destination_SVO[0 or 1]'))
avg = float(input('enter Average price'))
PREDICT = rt.predict([[Duration, Stops, par, avg, ruh, sov, d_par, d_ruh, d_sov]])[0]
PREDICT = PREDICT + ((avg / 100) * 5)
print(PREDICT, '$USD 5% Increase')
PREDICT = PREDICT + ((avg / 100) * 5)
print(PREDICT, '$USD')