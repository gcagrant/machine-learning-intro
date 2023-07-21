#!/usr/bin/env python
# coding: utf-8

# # Hypothesis
# 
# We can predict how many medals a country will win at the Olympics by using historical data.

# # The Data
# 
# A dataset of how many medals each country won at each Olympics.  Other data would also be nice (number of athletes, etc).

# In[248]:


import pandas as pd


# In[249]:


teams = pd.read_csv("teams.csv")


# In[250]:


teams


# In[251]:


teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]


# In[252]:


teams.corr()["medals"]


# In[253]:


import seaborn as sns


# In[254]:


sns.lmplot(x='athletes',y='medals',data=teams,fit_reg=True, ci=None) 


# In[255]:


sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None) 


# In[256]:


teams.plot.hist(y="medals")


# In[257]:


teams[teams.isnull().any(axis=1)].head(20)


# In[258]:


teams = teams.dropna()


# In[259]:


teams.shape


# In[260]:


train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()


# In[261]:


# About 80% of the data
train.shape


# In[262]:


# About 20% of the data
test.shape


# # Accuracy Metric
# 
# We'll use mean squared error.  This is a good default regression accuracy metric.  It's the average of squared differences between the actual results and your predictions.

# In[263]:


from sklearn.linear_model import LinearRegression

reg = LinearRegression()


# In[264]:


predictors = ["athletes", "prev_medals"]


# In[265]:


reg.fit(train[predictors], train["medals"])


# In[266]:


predictions = reg.predict(test[predictors])


# In[267]:


predictions.shape


# In[268]:


test["predictions"] = predictions


# In[269]:


test.loc[test["predictions"] < 0, "predictions"] = 0


# In[270]:


test["predictions"] = test["predictions"].round()


# In[271]:


from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test["medals"], test["predictions"])
error


# In[272]:


teams.describe()["medals"]


# In[273]:


test["predictions"] = predictions


# In[274]:


test[test["team"] == "USA"]


# In[275]:


test[test["team"] == "IND"]


# In[276]:


errors = (test["medals"] - predictions).abs()


# In[277]:


error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team 


# In[278]:


import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]


# In[279]:


error_ratio.plot.hist()


# In[280]:


error_ratio.sort_values()


# # Next steps
# 
# This model works well for countries which have a high medal count, and compete in a stable number of events annually.  For countries that get fewer medals, you'd want to build this model in a different way.
# 
# Some potential next steps:
# 
# * Add in some more predictors to the model, like `height`, `athletes`, or `age`.
# * Go back to the original, athlete-level data (`athlete_events.csv`), and try to compute some additional variables, like total number of years competing in the Olympics.
# * For countries with low medal counts, you can try modelling if individual athletes will win their event.  You can build event-specific models to predict if athletes will win their events.  Then you can add up the predicted medals for each athlete from each country.  This will give you the total predicted medal count for that country.
