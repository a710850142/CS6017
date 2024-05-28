import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Task 1: Import the data
df1 = pd.read_csv('realEstate1.csv')
df2 = pd.read_csv('realEstate2.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Task 2: Clean the data
df = df[(df['LstPrice'] >= 200000) & (df['LstPrice'] <= 1000000)]
columns_to_keep = ['Acres', 'Deck', 'GaragCap', 'Latitude', 'Longitude', 'LstPrice', 'Patio', 'PkgSpacs', 'PropType', 'SoldPrice', 'Taxes', 'TotBed', 'TotBth', 'TotSqf', 'YearBlt']
df = df[columns_to_keep]
df['TotSqf'] = df['TotSqf'].str.replace(',', '').astype(int)
df['Prop_Type_num'] = pd.factorize(df['PropType'])[0]
df = df[(df['Longitude'] != 0) & (df['Taxes'] < 20000)]

# Task 3: Exploratory data analysis
print("Dataset Description:")
print(df.describe())

plt.figure(figsize=(8, 6))
sns.countplot(x='PropType', data=df)
plt.title('Property Type Distribution')
plt.show()

# Exclude non-numeric columns when computing the correlation matrix
numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_df = df[numeric_columns]
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

columns_for_scatter = ['Acres', 'LstPrice', 'SoldPrice', 'Taxes', 'TotBed', 'TotBth', 'TotSqf', 'YearBlt']
sns.pairplot(df[columns_for_scatter])
plt.show()

# Task 4: Geospatial plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Longitude', y='Latitude', hue='SoldPrice', data=df, palette='viridis')
plt.title('House Location and Sold Price')
plt.show()

# Task 5: Simple Linear Regression
X = df[['LstPrice']]
y = df['SoldPrice']
model = LinearRegression()
model.fit(X, y)
print("Simple Linear Regression Results:")
print("R-squared:", model.score(X, y))
print("Coefficient (beta_1):", model.coef_[0])

plt.figure(figsize=(8, 6))
sns.regplot(x='LstPrice', y='SoldPrice', data=df)
plt.title('List Price vs. Sold Price')
plt.show()

# Task 6: Multilinear Regression
features = ['TotSqf', 'GaragCap', 'Latitude', 'Longitude']
X = df[features]
y = df['SoldPrice']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print("Multilinear Regression Results:")
print(model.summary())

# Task 7: Incorporating a categorical variable
model1 = sm.OLS(df['SoldPrice'], sm.add_constant(df['Prop_Type_num'])).fit()
print("Model 1 Results:")
print(model1.summary())

model2 = sm.OLS(df['SoldPrice'], sm.add_constant(df[['Prop_Type_num', 'TotSqf']])).fit()
print("Model 2 Results:")
print(model2.summary())

plt.figure(figsize=(8, 6))
sns.scatterplot(x='TotSqf', y='SoldPrice', hue='PropType', data=df)
plt.title('Total Square Footage vs. Sold Price')
plt.show()
