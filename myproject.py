#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
from sklearn import preprocessing
df_loc = pd.read_csv(r"E:\Medilab (1)\Medilab\survey3.csv")
#df_loc= df_loc.fillna() 
#print(len(df_loc))
df_loc["Occupation"] = df_loc["Occupation"].fillna('armed force')
#print(df_loc.head())
#print(len(df_loc))


# In[35]:


df_loc.drop(["Timestamp"], axis = 1, inplace = True)
df_loc.head()


# In[36]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
#print(df_loc['Occupation'])
#df_loc['Occupation']=df_loc["Occupation"].fillna("armed force", inplace = True)
#print(df_loc['Occupation'])
df_loc['Occupation']=df_loc['Occupation'].str.lower()
o_label_encoder = preprocessing.LabelEncoder() 
#occupation.unique()
df_loc['Occupation']= o_label_encoder.fit_transform(df_loc['Occupation']) 
df_loc['Occupation'].unique()


# In[38]:


g_label_encoder = preprocessing.LabelEncoder() 
df_loc['Gender']= g_label_encoder.fit_transform(df_loc['Gender']) 
df_loc['Gender'].unique()
#print(df_loc['Gender'])


# In[39]:


e_label_encoder = preprocessing.LabelEncoder() 
df_loc['Do you exercise?']=e_label_encoder.fit_transform(df_loc['Do you exercise?']) 
df_loc['Do you exercise?'].unique()
#print(df_loc['Do you exercise?'])


# In[40]:


s_label_encoder = preprocessing.LabelEncoder()
df_loc['Street']= s_label_encoder.fit_transform(df_loc['Street']) 
df_loc['Street'].unique()


# In[41]:


df_loc["City"] = df_loc["City"].fillna('Ernakulam')
df_loc['City']=df_loc['City'].str.lower()
df_loc['City']= label_encoder.fit_transform(df_loc['City']) 
df_loc['City'].unique()


# In[42]:


df_loc["Blood Group"]= df_loc["Blood Group"].replace("A+ve", "A+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("B +ve", "B+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("B+ve", "B+")
df_loc["Blood Group"]= df_loc["Blood Group"].replace("b positive", "B+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("B positive", "B+")
df_loc["Blood Group"]= df_loc["Blood Group"].replace("O positive", "O+")
df_loc["Blood Group"]= df_loc["Blood Group"].replace("A positive ", "A+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("O +", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("O +ve", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("B +", "B+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("O+ve", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("0+", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("O Positive", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("o positive", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("0+ ve", "O+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("AB+ve", "AB+")
df_loc['Blood Group']= df_loc['Blood Group'].replace("O postive", "O+")
df_loc["Blood Group"]= df_loc["Blood Group"].replace("A positive", "A+")
#df_loc['Blood Group']= df_loc['Blood Group'].replace("", "AB+")
#df_loc['Blood Group']= label_encoder.fit_transform(df_loc['Blood Group']) 
#df_loc['Blood Group'].unique()
#print(df_loc['Blood Group'])


# In[43]:


b_label_encoder = preprocessing.LabelEncoder()
df_loc['Blood Group']= b_label_encoder.fit_transform(df_loc['Blood Group']) 
df_loc['Blood Group'].unique()
#print(df_loc['Blood Group'])


# In[44]:


df_loc[' Are you suffering from any diseases ?']= df_loc[' Are you suffering from any diseases ?'].str.replace("yes", "Yes", case = True)
df_loc[' Are you suffering from any diseases ?']= df_loc[' Are you suffering from any diseases ?'].str.replace("no", "No", case = True)
#print(df_loc[' Are you suffering from any diseases ?'])


# In[45]:


df_loc[" Are you suffering from any diseases ?"].fillna("No", inplace = True)
df_loc[' Are you suffering from any diseases ?']= label_encoder.fit_transform(df_loc[' Are you suffering from any diseases ?']) 
df_loc[' Are you suffering from any diseases ?'].unique() 
#print(df_loc[' Are you suffering from any diseases ?'])


# In[46]:


df_loc["Details of the diseases"].fillna("No", inplace = True)
#print(df_loc["Details of the diseases"])
df_loc["Details of the diseases"]= label_encoder.fit_transform(df_loc["Details of the diseases"]) 
df_loc["Details of the diseases"].unique() 
#print(df_loc["Details of the diseases"])


# In[47]:


#df_loc["If yes, How often you exercise?"]= df_loc["If yes, How often you exercise?"].replace("NaN", "0")
et_label_encoder = preprocessing.LabelEncoder()
df_loc["If yes, How often you exercise?"].fillna("No", inplace = True)
df_loc['If yes, How often you exercise?']= et_label_encoder.fit_transform(df_loc['If yes, How often you exercise?']) 
df_loc['If yes, How often you exercise?'].unique()
#print(df_loc["If yes, How often you exercise?"])


# In[48]:


df_loc["How many hours do you sit and work or study?"].fillna("5", inplace = True)
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("none", "0")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("6 hours ", "6")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("04-May", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("07-Aug", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("1hr", "1")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("8½ hours", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("8 hrs", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("0.5", "1")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("2hrs", "2")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("It depends on the day's. Min 8 hours", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("9hrs", "9")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("Depends on work", "4")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("Yes", "4")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("24", "4")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("0", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("7-8", "8")
df_loc['How many hours do you sit and work or study?']= df_loc['How many hours do you sit and work or study?'].replace("4-5", "8")
#print(df_loc["How many hours do you sit and work or study?"])


# In[49]:


df_loc['Height'] = df_loc['Height'].str.replace(r"[a-zA-Z]+", "")
df_loc['Height'] = df_loc['Height'].replace("168 ","168")
df_loc['Height'] = df_loc['Height'].replace("5. 8","167")
df_loc['Height'] = df_loc['Height'].replace("5.9","167")
df_loc['Height'] = df_loc['Height'].replace("5.1","167")
df_loc['Height'] = df_loc['Height'].replace("155.","167")
df_loc['Height'] = df_loc['Height'].replace("154.","167")
df_loc['Height'] = df_loc['Height'].replace("5.3 ","167")
df_loc['Height'] = df_loc['Height'].replace("5.5 ","167")
df_loc['Height'] = df_loc['Height'].replace("5.7","167")
df_loc['Height'] = df_loc['Height'].replace("5.2","167")
df_loc['Height'] = df_loc['Height'].replace("5.6","167")

#print(df_loc['Height'])


# In[51]:


sl_label_encoder = preprocessing.LabelEncoder()
df_loc['How many hours you sleep a day?'] = df_loc['How many hours you sleep a day?'].replace("6","6 hour")
df_loc['How many hours you sleep a day?'] = df_loc['How many hours you sleep a day?'].replace("8","8 hour")
df_loc['How many hours you sleep a day?']= sl_label_encoder.fit_transform(df_loc['How many hours you sleep a day?']) 
df_loc['How many hours you sleep a day?'].unique()
#print(df_loc["How many hours you sleep a day?"])


# In[52]:


st_label_encoder = preprocessing.LabelEncoder()
df_loc["Do you sleep during the day time?"].fillna("No", inplace = True)
df_loc['Do you sleep during the day time?']= st_label_encoder.fit_transform(df_loc['Do you sleep during the day time?']) 
df_loc['Do you sleep during the day time?'].unique()
#print(df_loc["Do you sleep during the day time?"])


# In[53]:


df_loc["latitude"].fillna("10.1198N", inplace = True)
#print(df_loc["latitude"])


# In[54]:


df_loc['latitude']=df_loc['latitude'].str[:-1]

#df_loc['latitude']= df_loc['latitude'].map(lambda x: re.sub(r'\W+', '', x))
df_loc['latitude']=df_loc['latitude'].str.replace('°', '')
df_loc['latitude']=df_loc['latitude'].str.replace('N', '')
df_loc['latitude']=df_loc['latitude'].str.replace('Â° ', '')
#df_loc['latitude']= df_loc['latitude'].str.rstrip()
#print(df_loc['latitude'])
#re.sub('[^0-9]+', '', ) 


# In[55]:


df_loc['longitude']=df_loc['longitude'].str[:-1]
df_loc['longitude']=df_loc['longitude'].str.replace('°', '')
df_loc['longitude']=df_loc['longitude'].str.replace('E', '')
#df_loc['longitude'].fillna(df_loc['longitude'].mean())
df_loc["longitude"].fillna("76.4", inplace = True)
#df_loc['longitude']= df_loc['longitude'].str.rstrip()
#print(df_loc['longitude'])
#df_loc['longitude']= df_loc['longitude'].map(lambda x: re.sub(r'\W+', '', x))


# In[56]:


df_new=df_loc[['Age','Gender','Street','Occupation','How many hours do you sit and work or study?','Blood Group','Do you exercise?','If yes, How often you exercise?','How many hours you sleep a day?','Do you sleep during the day time?','Height','Weight','latitude','longitude']]
df_new.to_csv(r'E:\Medilab (1)\Medilab\survey.csv')
#print(df_new)


# In[57]:


df_new1=df_loc[[' Are you suffering from any diseases ?']]
df_new1.to_csv(r'E:\Medilab (1)\Medilab\survey1.csv')
#print(df_new1)


# In[58]:


import numpy as np
#from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler
#df_new.fillna(df_new.mean())
#X = StandardScaler().fit_transform(df_new)


# In[59]:


# Compute DBSCAN
#db = DBSCAN(eps=.8, min_samples=10).fit(X)
#normalize the data
min_max_scalar = preprocessing.MinMaxScaler()
x_scaled = min_max_scalar.fit_transform(df_new)
df_norm = pd.DataFrame(x_scaled)
db=Birch(branching_factor=50, n_clusters=5, threshold=25,compute_labels=True)
db.fit_transform(df_new)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
cluster_centers = db.subcluster_centers_


# In[60]:


import matplotlib.pyplot as plt
cluster_centers = db.subcluster_centers_
labels = db.labels_
fig = plt.figure(figsize=(15, 15), dpi=200)
ax = fig.add_subplot(111)
ax.set_title("Cluter centers on 2 Component PCA in Disease Dataset on Birch")
for x,y,lab in zip(cluster_centers[:, 0],cluster_centers[:, 1],labels):
        ax.scatter(x,y,label=lab, s= 250)
colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired 
colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]       
for t,j1 in enumerate(ax.collections):
    j1.set_color(colorst[t])


# In[61]:


# Number of clusters in labels, ignoring noise if present.
from sklearn import metrics
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(x_scaled ,labels))


# In[62]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_new, df_new1, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


# In[63]:


def predictDisease(age,gender,street,occupation,worktime,bloodgroup,exercise,exercisetime,sleeptime,daysleep,height,weight,latitude,longitude):         
        gender_t = g_label_encoder.fit_transform([gender])
        street_t=s_label_encoder.fit_transform([street])
        occupation_t=o_label_encoder.fit_transform([occupation])
        bloodgroup_t=b_label_encoder.fit_transform([bloodgroup])
        exercise_t=e_label_encoder.fit_transform([exercise])
        exercisetime_t=et_label_encoder.fit_transform([exercisetime])
        sleeptime_t=sl_label_encoder.fit_transform([sleeptime])
        daysleep_t=st_label_encoder.fit_transform([daysleep])
        X_train=[age,gender_t,street_t,occupation_t,worktime,bloodgroup_t,exercise_t,exercisetime_t,sleeptime_t,daysleep_t,height,weight,latitude,longitude]
        X_pred=clf.predict([X_train])
        
        print(X_pred)
        if X_pred>0:
             return 'chance of disease'
        else:
             return 'No chance of disese'


# In[64]:


predictDisease(30,"Male","nedungapra","it profesional",6,"B+","No",0,6,"Yes",167,83,10.0460,76.7221)

