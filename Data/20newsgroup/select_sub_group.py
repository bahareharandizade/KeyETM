import pandas as pd
import os

def select_subset(df,col,cond_list):
  subset_df = df[df[col].isin(cond_list)]
  return subset_df


cond_list = ["talk.politics.guns","soc.religion.christian","sci.electronics",'talk.politics.mideast',"sci.space","sci.med"]
df = pd.read_csv('20newsgroup.csv')
df = select_subset(df,"target",cond_list)
df.to_csv('newsgroup_subset.csv')



