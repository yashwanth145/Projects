import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('dataset.csv')
df=df.head(100)
df=df[['Name','Age','Department','Salary','Performance_Score','Experience','Location']]

#Age vs Salary
sns.kdeplot(data=df,x='Age',y='Salary',color='green')
plt.xlabel='Age'
plt.ylabel='Salary'
plt.title='Employees performance as per their age'
plt.show()

#Age vs experiance
sns.kdeplot(data=df,x='Age',y='Experience',color='blue')
plt.xlabel='Age'
plt.ylabel='Experience'
plt.title='Employees performance as per their Experiance'
plt.show()
