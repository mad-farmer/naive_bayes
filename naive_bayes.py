import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("heart.csv") #Import csv file

#%% Data Exploration
df.isnull().any() #Check for null data
df.head() #First 5 rows of our data
#%%
sns.countplot(x="target", data=df)
#target: have disease or not (1=yes, 0=no
plt.show()
#%%
sns.countplot(x='sex', data=df)
plt.xlabel("female=0  male=1")
plt.show()
#%%
pd.crosstab(df.age,df.target).plot(
        kind="bar",figsize=(15,7))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()
#%%
pd.crosstab(df.sex,df.target).plot(
        kind="bar",figsize=(15,7))
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
#%%
plt.scatter(x=df.age[df.target==1], 
            y=df.thalach[(df.target==1)])
plt.scatter(x=df.age[df.target==0], 
            y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


#%%Data preprocessing
y=df["target"].values
#target: have disease or not (1=yes, 0=no)
x=df.drop(["target"],axis=1).values 

sc=StandardScaler()
x = sc.fit_transform(x)  
#Scaling features

x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.2,random_state=42)
#train test split


#%%
gb=GaussianNB()
gb.fit(x_train,y_train) 
#fitting
y_pred=gb.predict(x_test)
#classification test values


#%%
print("Accuracy:",gb.score(x_test,y_test))
#Accuracy: 0.8688524590163934



#%%


























#%%