#!/usr/bin/env python
# coding: utf-8

# In[284]:


import numpy as np;
import pandas as pd;
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
##csvFilePath = "link to csv here"; # I didn't know if I can share the csv file on git hub. That's why I left this to fill with path. 
df = pd.read_csv(csvFilePath, sep = '\",\"', header = 0, engine = 'python');

percentTrainData = 0.9;
decision = 0;


# In[285]:


#Function which prepare data from csv to analise (changing GEO to GEOx and GEOy, delating chars(,"),
#changing sector to numeric values with one hot encoding)
def preapareDataFrameForAnalise(df):
    df[['\"ID\"','\"GEO\"']] = df["\"ID,\"\"GEO\""].str.split(',\"', expand=True);
    df.pop("\"ID,\"\"GEO\"");
    df[['xGEO','yGEO']] = df["\"GEO\""].str.split(',', expand=True);
    df.pop("\"GEO\"");
    df.columns = df.columns.str.replace('\"', ''); 
    for column in df.columns:
        df[column] = df[column].str.replace('\"', '');
    oe_style = OneHotEncoder();
    obj_df= df.select_dtypes(include=['object']).copy();
    oe_results = oe_style.fit_transform(obj_df[["SECTOR"]]);
    sectorCoded = pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_);
    df = df.join(sectorCoded);
    df.pop("SECTOR");
    df.pop("ID");
    for column in df.columns:
        df[column] = pd.to_numeric(df[column],errors='coerce');
    return df;

#Function which handle NaN values - if sum of NaN in column is higher than 30% of sum of rows in column, column is deleted
# otherwise NaN values are replaced with median (TODO)
def handleNaNValues(df):
    for column in df.columns:
        nancounter = df[column].isnull().sum();
        if(nancounter > 0):
            if(nancounter/len(df) > 0.3):
                df.pop(column)
                print("Column " + column + " was deleted because there were too many NaN values - " + str( 100 * nancounter/len(df)) + "% ");
            #else:
                #TODO - change NaN to calculated median value
    return df;

def makeConfusionMatrix(x, y):
    cm = confusion_matrix(y, model.predict(x))
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual value');
    plt.xlabel('Predicted value');
    all_sample_title = 'Accuracy Score: {0:.2f}%'.format((model.score(x, y)*100))
    plt.title(all_sample_title, size = 15);

def makeModel(df):
    dfProbe = df.head(round(len(df)*percentTrainData));
    model = LogisticRegression(solver='liblinear', random_state=0);
    y = dfProbe.pop("FLAG");
    x = dfProbe;
    model.fit(x, y);
    makeConfusionMatrix(x,y);
    print(classification_report(y, model.predict(x)))
    return model;

def checkHowGoodModelIs(model, df):
    goodAnalyseCounter = 0;
    testData = df.tail((len(df) - round(len(df)*percentTrainData)));
    dfFlag = testData.pop("FLAG");
    result = model.predict(testData);
    print(str(model.score(testData, dfFlag)*100) + "% of result was up to expecations");
    print(classification_report(dfFlag, result))
    makeConfusionMatrix(testData,dfFlag);
  
  
def checkData(model, df): 
    if("FLAG" in df.columns):
        df.pop("FLAG");
    for i in range(0,len(df)):
        print("Company data:")
        print("Company id:"+ str(df.iloc[[i]].index.values[0]+1))
        print("Chance for company bankruptcy: " + str(model.predict_proba(df)[i][0]))
        print("Chance for company survive: " + str(model.predict_proba(df)[i][1]))
    

    


# In[286]:



df = preapareDataFrameForAnalise(df);


# In[287]:


df = handleNaNValues(df);


# In[288]:


model = makeModel(df);


# In[289]:


checkHowGoodModelIs(model, df);


# In[ ]:




