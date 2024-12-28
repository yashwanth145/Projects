import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Boston-house-price-data.csv')
df=df.head(200)
df=df[['Crime_rate','Residential_zone','Charles_river','Nitrix_oxide','average_number_of_rooms','Age','Distance','Tax','Black_residents','Low_status']]


#Crime rates vs number of rooms
sns.kdeplot(data=df,x='Crime_rate',y='average_number_of_rooms',color='red')
plt.title('Crime rate as per average rooms available')
plt.xlabel('Crime_rate')
plt.ylabel('average_number_of_rooms')
plt.show()


#average number of rooms vs distance to IT hub
sns.kdeplot(data=df,x='average_number_of_rooms',y='Distance',color='blue')
plt.title('Nearer distances for IT techies')
plt.xlabel('average_number_of_rooms')
plt.ylabel('Distance')
plt.show()


#average number of rooms vs Tax
sns.kdeplot(data=df,x='average_number_of_rooms',y='Tax',color='green')
plt.title('Tax levied on the people')
plt.xlabel('average_number_of_rooms')
plt.ylabel('Tax')
plt.show()


#average number of rooms vs Low status residents
sns.kdeplot(data=df,x='average_number_of_rooms',y='Low_status',color='red')
plt.title('Accomadation according to your earnings')
plt.xlabel('average_number_of_rooms')
plt.ylabel('Low_status')
plt.show()