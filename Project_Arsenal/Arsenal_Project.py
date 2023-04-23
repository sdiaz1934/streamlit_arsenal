#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing database and libraries

import warnings

from IPython import get_ipython

warnings.filterwarnings('ignore')


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
# get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.models import CheckboxGroup, CustomJS
from bokeh.layouts import row
from bokeh.models import HoverTool

from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Plotting the graphs in the Jupyter notebook

output_notebook()

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)

df=pd.read_csv('Project_Arsenal/df_full_premierleague.csv')
# df.head(10)


# In[2]:


# df.info()


# In[3]:


# Removing unnecessary columns

columns_types=df.dtypes

# print(columns_types)

        
# print((df['Unnamed: 0']!=df.index).sum())

# The column 'Unnamed: 0' is identical to the index.
# Since the index is already integrated in the table, there is no need to store it in a separate column.
# The column 'link_match' represents links that directs you to the match information. 
# There is no need to keep this column since most of the data is already stored in other columns
# The columns result_full, result_ht, goal_home_ft, goal_away_ft, sg_match_ft, goal_home_ht, goal_away_ht, sg_match_ht
# represent important information about the results of the matches and this could be quite important to determine the performance of a team.
# Nevertheless, this columns represent somehow the same data and could be summarize in fewer columns.
# For the purpose of this analysis, the information will be summarize in two new columns which are goals_home and goals_away.
# Additionally, to study further the performance of a team we need information related to the offensive and defensive indicators. 
# There is too many information on this aspect which is summarize in the half and 
# full time results of a match which can be not considered since we need the actual result of the match.
# Therefore, the following columns will not be taken into account:

col_list = df.columns.values.tolist()

columns_to_delete=["Unnamed: 0","link_match","result_full", "result_ht"]
columns_to_delete.extend(col_list[32:])
columns_to_delete.extend(([' ',' ',' ',' '])) #To reshape the list to have an even number(90)

# convert list to numpy array
np_array=np.asarray(columns_to_delete)
# reshape array into 18 rows x 5 columns
reshaped_array = np_array.reshape(18, 5)

#now construct a table
deleted=pd.DataFrame(reshaped_array, columns=['columns','to','delete','  ',''])

# deleting the ' ' created
columns_to_delete.remove(' ')
columns_to_delete.remove(' ')
columns_to_delete.remove(' ')
columns_to_delete.remove(' ')

deleted


# In[4]:


df['home_goals'] = df['result_full'].str.split('-').str[0].astype('int')


df['away_goals'] = df['result_full'].str.split('-').str[1].astype('int')

df['points_home']=0

df['points_away']=0

df['winner']='' #This column will tell if the winner team is the home team or the away team(or if it was a draw)

df['W']=0
df['L']=0
df['D']=0

for ind in df.index:
    if df['home_goals'][ind]>df['away_goals'][ind]:
        df['points_home'][ind]+=3
        df['winner'][ind]='H'
        df['W'][ind]+=1

       
    elif df['home_goals'][ind]<df['away_goals'][ind]:
        df['points_away'][ind]+=3
        df['winner'][ind]='A'
        df['L'][ind]+=1

      
    else:
        df['points_home'][ind]+=1
        df['points_away'][ind]+=1
        df['winner'][ind]='D'
        df['D'][ind]+=1

     
        
    
df=df.drop(columns_to_delete,axis=1)

# df.info()


# In[5]:


# df.head(100)


# In[6]:


# Identifying and/or deleting duplicate values and NaN management

duplicate_values=df.duplicated().sum()
empty_cells=df.isnull().sum()

# for ind,val in empty_cells.items():
#     if empty_cells[ind]==0:
#         empty_cells.drop(labels=[ind],axis=0,inplace=True)
        
# print(f'The database has {duplicate_values} duplicated values\n\n The database has {empty_cells} missing values')
# print(f'The number of matches per Season are:\n\n {df["season"].value_counts()}')

# Formating Date

df["date"]=pd.to_datetime(df["date"], dayfirst=True)


# In[7]:


# In this section we will adjust the data later take a look at the performance of the performance per season of each team.


# New dataframe to obtain performance per season
# Grouping points,goals,passes, average possession, shot, and shots on target per season and then merging the data

def season_total(df, home_field, away_field,name_1,name_2):
    home_grouped = df.groupby(['season','home_team'], as_index = False).agg({home_field:'sum'})
    away_grouped = df.groupby(['season','away_team'], as_index = False).agg({away_field:'sum'})
    grouped = home_grouped.merge(away_grouped,how="inner",on="season")
    grouped=grouped.rename(columns={home_field: name_1, away_field: name_2})
    return grouped

def season_avg(df, home_field, away_field,name_1,name_2):
    home_grouped = df.groupby(['season','home_team'], as_index = False).agg({home_field:'mean'})
    away_grouped = df.groupby(['season','away_team'], as_index = False).agg({away_field:'mean'})
    grouped = home_grouped.merge(away_grouped,how="inner",on="season")
    grouped=grouped.rename(columns={home_field: name_1, away_field: name_2})
    return grouped


# In[8]:


# Points per season
points_per_season=season_total(df, 'points_home', 'points_away','Points(home)','Points(away)')

# Goals scored per season
goals_scored_per_season=season_total(df, 'home_goals', 'away_goals','Goals scored(home)','Goals scored(away)')

# Goals conceded per season
goals_conceded_per_season=season_total(df, 'away_goals', 'home_goals','Goals conceded(home)','Goals conceded(away)')

# Passes per season
passes_per_season=season_total(df, 'home_passes', 'away_passes','Passes(home)','Passes(away)')

# Shots per season
shots_per_season=season_total(df, 'home_shots', 'away_shots','Shots(home)','Shots(away)')

# Shots on target per season
shots_on_target_per_season=season_total(df, 'home_shots_on_target', 'away_shots_on_target',
                                        'Shots on target(home)','Shots on target(away)')


# In[9]:


# Average points scored per season
avg_points_per_season=season_avg(df, 'points_home', 'points_away','Avg Points(home)','Avg Points(away)')

# Average goals scored per season
avg_goals_scored_per_season=season_avg(df, 'home_goals', 'away_goals','Avg Goals scored(home)','Avg Goals scored(away)')

# Average goals conceded per season
avg_goals_conceded_per_season=season_avg(df, 'away_goals','home_goals','Avg Goals conceded(home)','Avg Goals conceded(away)')

# Average Passes per season
avg_passes_per_season=season_avg(df, 'home_passes', 'away_passes','Avg Passes(home)','Avg Passes(away)')

# Average Possession
possession_per_season=season_avg(df, 'home_possession', 'away_possession','Avg Possession(home)','Avg Possession(away)')

# Average Shots per season
avg_shots_per_season=season_avg(df, 'home_shots', 'away_shots','Avg Shots(home)','Avg Shots(away)')

# Average Shots on target per season
avg_shots_on_target_per_season=season_avg(df, 'home_shots_on_target', 'away_shots_on_target',
                                          'Avg Shots on target(home)','Avg Shots on target(away)')


# In[10]:


# Merging the databases

performance_per_season=points_per_season.merge(goals_scored_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(goals_conceded_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(passes_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(possession_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(shots_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(shots_on_target_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(avg_points_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(avg_goals_scored_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(avg_goals_conceded_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(avg_passes_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(avg_shots_per_season,how="inner",
                                               on=["season",'home_team','away_team']).merge(avg_shots_on_target_per_season,how="inner",
                                               on=["season",'home_team','away_team'])



for column in performance_per_season.columns:
    type_column=performance_per_season[column].dtypes
    if type_column=='float64':
        performance_per_season[column]=performance_per_season[column].round(2)
        

# The new database will have all matches from 2010 to 2020. We will delete then all rows where the home team is different from the away team.
# This way we will have then the information per team for each season without additional rows that could lead to a wrong analysis

for ind,row in performance_per_season.iterrows():
    if row['home_team']!=row['away_team']:
        performance_per_season.drop(ind,axis=0,inplace=True)
        

# We will delete the results of the season 20/21.
# The information of this season is partial(half of the season) and we are analyzing the results at the end of each season
# This will only be done on the analysis of the Big 6. 
# The information of this season will be taken into account for the analysis of the performance of Arsenal

for ind,row in performance_per_season.iterrows():
    if row['season']=='20/21':
        performance_per_season.drop(ind,axis=0,inplace=True)
        
# We will calculate the total of points,goals,passes, average possession, shot, shots on target 
# and table standing for each team
                         
performance_per_season['Total_points']=performance_per_season['Points(home)']+performance_per_season['Points(away)']

performance_per_season['Total_goals_scored']=performance_per_season['Goals scored(home)']+performance_per_season['Goals scored(away)']

performance_per_season['Total_goals_conceded']=(performance_per_season['Goals conceded(home)']
                                                +performance_per_season['Goals conceded(away)'])

performance_per_season['Diff']=performance_per_season['Total_goals_scored']-performance_per_season['Total_goals_conceded']

performance_per_season['Total_passes']=performance_per_season['Passes(home)']+performance_per_season['Passes(away)']

performance_per_season['Average_possession']=((performance_per_season['Avg Possession(home)']
                                              +performance_per_season['Avg Possession(away)'])/2).round(2)

performance_per_season['Total_shots']=performance_per_season['Shots(home)']+performance_per_season['Shots(away)']

performance_per_season['Total_shots_on_target']=(performance_per_season['Shots on target(home)']+
                                                performance_per_season['Shots on target(away)'])

performance_per_season['Avg_points']=(performance_per_season['Avg Points(home)']
                                     +performance_per_season['Avg Points(away)'])/2

performance_per_season['Avg_goals_scored']=(performance_per_season['Avg Goals scored(home)']
                                     +performance_per_season['Avg Goals scored(away)'])/2

performance_per_season['Avg_goals_conceded']=(performance_per_season['Avg Goals conceded(home)']
                                     +performance_per_season['Avg Goals conceded(away)'])/2

performance_per_season['Avg_passes']=(performance_per_season['Avg Passes(home)']
                                      +performance_per_season['Avg Passes(away)'])/2

performance_per_season['Avg_shots']=(performance_per_season['Avg Shots(home)']
                                     +performance_per_season['Avg Shots(away)'])/2

performance_per_season['Avg_shots_on_target']=(performance_per_season['Avg Shots on target(home)']+
                                                performance_per_season['Avg Shots on target(away)'])/2

# Table Standing

team_count = performance_per_season.groupby(['season'])['home_team'].count().tolist()
rank_column = []

for i in team_count :
    j = list(range(1,i+1,1))
    rank_column += j

performance_per_season = performance_per_season.sort_values(['season', 'Total_points', 'Diff'], ascending=[True, False, False])
performance_per_season['Table_standing'] = rank_column
performance_per_season.reset_index(inplace=True)

performance_per_season.drop(['away_team','index'],axis=1,inplace=True)

columns_to_replace={'home_team': 'Team','season':'Season'}

performance_per_season.rename(columns=columns_to_replace, inplace=True)

performance_per_season


# In[11]:


# The champions of each season will serve as a guide to analyze why Arsenal has not been able to win the premier league 
premier_league_champions=performance_per_season[(performance_per_season['Table_standing']==1)]
champions_vs_arsenal=premier_league_champions['Team'].unique().tolist()
champions_vs_arsenal.append('Arsenal')
champions_vs_arsenal


# In[12]:


# Stastistics of the best teams
best_teams=performance_per_season[(performance_per_season['Table_standing']<=4)]
list_best_teams=best_teams['Team'].unique().tolist()
list_best_teams


# In[13]:


# Stastistics of the champions
champions_teams=performance_per_season[(performance_per_season['Table_standing']==1)]
list_champions_teams=champions_teams['Team'].unique().tolist()
champions_teams


# In[14]:


# Since the column 'Season' is a string then we will create a new column to name the Seasons as a sequence of numbers 

list_teams=performance_per_season['Team'].unique().tolist()
performance_per_season['Season_number']=0
seasons=performance_per_season['Season'].unique().tolist()
for i in range(len(seasons)):
    performance_per_season['Season_number'][performance_per_season['Season']==seasons[i]]=i


# In[15]:


# This is a function to create interactive graphs using Bokeh. We can decide if we use it or not

def graph_lines(data,variable):
    from bokeh.models import MultiChoice
    from bokeh.layouts import row
    import random
    
    tooltips = [("Team", "@Team"), 
            ("Season", "@Season"),
           (variable,"@"+variable)]
    
    L = '0123456789ABCDEF'
    fig = figure(width=600, height=600)
    line_renderer = []
    names = data['Team'].unique().tolist()
    seasons=data['Season'].unique().tolist()
    
    for name in names:
        line_renderer.append(
            fig.line(x = 'Season_number',
                     y = variable,
                     color=(random.randrange(255), random.randrange(255), random.randrange(255)),
#                      color='#' + ''.join(random.choices('0123456789ABCDEF', k=6)),
                     name =name,
                     source=performance_per_season[performance_per_season['Team']==name]
                    )
        )
        
    for line in line_renderer:
        line.visible=False
            
    checkbox1 = CheckboxGroup(labels=names[:12], active=[],
                              width=100,margin=25)
    callback1 = CustomJS(args=dict(lines=line_renderer[:12],checkbox1=checkbox1),
                             code="""
                             for(var i=0; i<lines.length; i++){
                             lines[i].visible = checkbox1.active.includes(i);
                             }
                             """
                            )
        
    checkbox1.js_on_change('active', callback1)
        
    checkbox2 = CheckboxGroup(labels=names[12:24], active=[],
                              width=100,margin=(25,50))
    callback2 = CustomJS(args=dict(lines=line_renderer[12:24],checkbox2=checkbox2),
                             code="""
                             for(var i=0; i<lines.length; i++){
                             lines[i].visible = checkbox2.active.includes(i);
                             }
                             """
                            )
        
    checkbox2.js_on_change('active', callback2)
        
    checkbox3 = CheckboxGroup(labels=names[24:], active=[],
                              width=100,margin=25)
    callback3 = CustomJS(args=dict(lines=line_renderer[24:],checkbox3=checkbox3),
                             code="""
                             for(var i=0; i<lines.length; i++){
                             lines[i].visible = checkbox3.active.includes(i);
                             }
                             """
                            )
        
    checkbox3.js_on_change('active', callback3)
        
    multi_choice = MultiChoice(value=["Arsenal"], options=names)
    callback = CustomJS(args=dict(lines=line_renderer, multi_choice=multi_choice),
                            code="""
                            var selected_vals = multi_choice.value;
                            var index_check = [];
                            for (var i = 0; i < lines.length; i++) {
                            index_check[i]=selected_vals.indexOf(lines[i]);
                            if ((index_check[i])>=0){
                            lines[i].visible = true
                            }
                            else{
                            lines[i].visible=false;
                            }
                            """
                           )
        
    multi_choice.js_on_change("value", callback)
        
    fig.xaxis.ticker = [0,1,2,3,4,5,6,7,8,9]
    fig.xaxis.major_label_overrides = {0: '10/11', 1: '11/12', 2: '12/13',3:'13/14',
                                  4:'14/15',5:'15/16',6:'16/17',7:'17/18',
                                  8:'18/19',9:'19/20'}
        
    if variable=='Table_standing':
        fig.y_range.flipped = True
    
    fig.xaxis.axis_label = 'Season'
    fig.yaxis.axis_label = variable
    hover = HoverTool(tooltips = tooltips)
    fig.add_tools(hover)
    layout = row(fig,checkbox1,checkbox2,checkbox3)
    show(layout)
    return layout

def graph_seaborn(data,x,y,title):
    fig, ax = plt.subplots(figsize=(12,12))
    sns.lineplot(data=data, x=x,y=y,hue='Team')
    sns.set(rc={"figure.figsize":(10, 10)}) #width=5, #height=6

    # # specify axis labels
    # ax.set(xlabel='Season',
    #        ylabel=y,
    #        title=y+' '+'by season',
    #        ylim=(0,15)) #width=5, #height=6

    # Changing tick labels
    ax.set_xticks(range(len(seasons))) # <--- set the ticks first
    ax.set_xticklabels(['10/11','11/12','12/13','13/14','14/15',
                        '15/16','16/17','17/18','18/19','19/20'])


    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    if y=='Table_standing':
        # specify axis labels
        ax.set(xlabel='Season',
               ylabel=y,
               title=title,
               ylim=(0, 15))  # width=5, #height=6

        ax.invert_yaxis()
        left, bottom, width, height = (0, 1, 9, 3)
        rect=mpatches.Rectangle((left,bottom),width,height,
                                fill=True,
                                alpha=0.1,
                                facecolor="red")
        plt.gca().add_patch(rect)
    else:
        # specify axis labels
        ax.set(xlabel='Season',
               ylabel=y,
               title=title)  # width=5, #height=6

    return fig

def graph_seaborndfdf(data,x,y):
    fig, ax = plt.subplots(figsize=(12,12))
    sns.lineplot(data=data, x=x,y=y,hue='Team')
    sns.set(rc={"figure.figsize":(10, 10)}) #width=5, #height=6

    # specify axis labels
    ax.set(xlabel='Season',
           ylabel=y,
           title=y+' '+'by season',
           ylim=(0,15)) #width=5, #height=6

    # Changing tick labels
    ax.set_xticks(range(len(seasons))) # <--- set the ticks first
    ax.set_xticklabels(['10/11','11/12','12/13','13/14','14/15',
                        '15/16','16/17','17/18','18/19','19/20'])


    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    return fig


# In[16]:


filtered_data = performance_per_season.loc[performance_per_season['Team'].isin(list_best_teams)]

filtered_data = filtered_data[['Season_number','Team','Table_standing','Total_points','Total_goals_scored','Total_goals_conceded']]

filtered_data.groupby("Team")['Total_points','Total_goals_scored','Total_goals_conceded'].mean(numeric_only=True)

# filtered_data.groupby("Team").mean(numeric_only=True)


# In[17]:


champions_data = performance_per_season.loc[performance_per_season['Team'].isin(list_champions_teams)]

champions_data = champions_data[['Season_number','Team','Table_standing','Total_points','Total_goals_scored','Total_goals_conceded']]

# champions_data.groupby("Team")['Total_points','Total_goals_scored','Total_goals_conceded'].mean(numeric_only=True)

# champions_data.groupby("Team").head()

champions_data[champions_data['Table_standing']==1]


# In[18]:


filtered_data[filtered_data['Team']=='Arsenal']


# In[19]:


# # Table Standing
# ax=sns.lineplot(data=filtered_data, x='Season_number',y='Table_standing',hue='Team')
#
# sns.set(rc={"figure.figsize":(10, 10)}) #width=5, #height=6
#
# #specfiy axis labels
# ax.set(xlabel='Season',
#        ylabel='Table Standing',
#        title='Table standing by season',
#        ylim=(0,15)) #width=5, #height=6
#
# # Changing tick labels
#
# ax.set_xticks(range(len(seasons))) # <--- set the ticks first
# ax.set_xticklabels(['10/11','11/12','12/13','13/14','14/15',
#                    '15/16','16/17','17/18','18/19','19/20'])
#
# ax.invert_yaxis()
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#
# # Rectangule
# left, bottom, width, height = (0, 1, 9, 3)
# rect=mpatches.Rectangle((left,bottom),width,height,
# #                         fill=True,
#                         alpha=0.1,
#                        facecolor="red")
# plt.gca().add_patch(rect)
#
# #display barplot
# plt.show();


# In[20]:


# # Goals scored per season
# ax2=sns.lineplot(data=filtered_data, x='Season_number',y='Total_goals_scored',hue='Team')
#
# sns.set(rc={"figure.figsize":(10, 10)}) #width=5, #height=6
#
# #specfiy axis labels
# ax2.set(xlabel='Season',
#        ylabel='Goals scored',
#        title='Goals scored by season') #width=5, #height=6
#
# # Changing tick labels
#
# ax2.set_xticks(range(len(seasons))) # <--- set the ticks first
# ax2.set_xticklabels(['10/11','11/12','12/13','13/14','14/15',
#                    '15/16','16/17','17/18','18/19','19/20'])
#
#
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#
# #display barplot
# plt.show();


# In[21]:


# # Goals scored per season
# ax3=sns.lineplot(data=filtered_data, x='Season_number',y='Total_goals_conceded',hue='Team')
#
# sns.set(rc={"figure.figsize":(10, 10)}) #width=5, #height=6
#
# #specfiy axis labels
# ax3.set(xlabel='Season',
#        ylabel='Goals conceded',
#        title='Goals conceded by season') #width=5, #height=6
#
# # Changing tick labels
#
# ax3.set_xticks(range(len(seasons))) # <--- set the ticks first
# ax3.set_xticklabels(['10/11','11/12','12/13','13/14','14/15',
#                    '15/16','16/17','17/18','18/19','19/20'])
#
#
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#
# #display barplot
# plt.show();


# In[22]:


# # Total Points
# ax=sns.lineplot(data=filtered_data[filtered_data['Team']=='Arsenal'], x='Season_number',y='Total_points',hue='Team')
#
# sns.set(rc={"figure.figsize":(6, 5)}) #width=5, #height=6
#
# #specfiy axis labels
# ax.set(xlabel='Season',
#        ylabel='Total points',
#        title='Total points by season') #width=5, #height=6
#
# # Changing tick labels
#
# ax.set_xticks(range(len(seasons))) # <--- set the ticks first
# ax.set_xticklabels(['10/11','11/12','12/13','13/14','14/15',
#                    '15/16','16/17','17/18','18/19','19/20'])
#
#
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#
# # Rectangule
# left, bottom, width, height = (0, 1, 9, 3)
# rect=mpatches.Rectangle((left,bottom),width,height,
# #                         fill=True,
#                         alpha=0.1,
#                        facecolor="red")
# plt.gca().add_patch(rect)
#
# #display barplot
# plt.show();


# In[23]:


# Total points per season
# graph_lines(data=performance_per_season,variable='Total_points')


# In[24]:


# Goals scored per season
# graph_lines(data=performance_per_season,variable='Total_goals_scored')


# In[25]:


# Goals conceded per season
# graph_lines(data=performance_per_season,variable='Total_goals_conceded')


# In[26]:


# Table Standing
# graph_lines(data=performance_per_season,variable='Table_standing')


# In[27]:


# Distribution of games won for champions teams and Arsenal

n=1
fig = plt.figure(figsize=(20,30))
for i in champions_vs_arsenal:
    plt.subplot(8,3,n)
    n+=1
    plt.bar(1,df[(df['home_team']==i) | (df['away_team']==i)]['W'].sum(),width=0.15,label='Matches won')
    plt.bar(1.2,df[(df['home_team']==i) | (df['away_team']==i)]['L'].sum(),width=0.15,label='Matches lost')
    plt.bar(1.4,df[(df['home_team']==i) | (df['away_team']==i)]['D'].sum(),width=0.15,label='Draw')
    plt.title(i)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    plt.ylabel('Number of games')


plt.legend(loc='center left',bbox_to_anchor=(1, 0.1))    
plt.show();


# In[28]:


points=[]
goals_scored=[]
goals_conceded=[] 
passes=[] 
shots=[] 
shots_target=[] 

for i in champions_vs_arsenal:
    avg_points=performance_per_season[performance_per_season['Team']==i]['Avg_points'].mean()
    avg_points=round(avg_points,2)
    avg_goals_scored=performance_per_season[performance_per_season['Team']==i]['Avg_goals_scored'].mean()
    avg_goals_scored=round(avg_goals_scored,2)
    avg_goals_conceded=performance_per_season[performance_per_season['Team']==i]['Avg_goals_conceded'].mean()
    avg_goals_conceded=round(avg_goals_conceded,2)
    avg_passes=performance_per_season[performance_per_season['Team']==i]['Avg_passes'].mean()
    avg_passes=round(avg_passes,2)
    avg_shots=performance_per_season[performance_per_season['Team']==i]['Avg_shots'].mean()
    avg_shots=round(avg_shots,2)
    avg_shots_on_target=performance_per_season[performance_per_season['Team']==i]['Avg_shots_on_target'].mean()
    avg_shots_on_target=round(avg_shots_on_target,2)
    points.append(avg_goals_scored)
    goals_scored.append(avg_goals_scored)
    goals_conceded.append(avg_goals_conceded)
    passes.append(avg_passes)
    shots.append(avg_shots)
    shots_target.append(avg_shots_on_target)

# Normalization of the lists and creating Dataframe
points = 5*(points / np.linalg.norm(points))
goals_scored = 5*(goals_scored / np.linalg.norm(goals_scored))
goals_conceded = 5*(goals_conceded / np.linalg.norm(goals_conceded))
passes = 5*(passes / np.linalg.norm(passes))
shots = 5*(shots / np.linalg.norm(shots))
shots_target = 5*(shots_target / np.linalg.norm(shots_target))


# In[29]:


# Set data and Figure
df_teams = pd.DataFrame({
'Team': champions_vs_arsenal,
'Points': points,
'Goals scored': goals_scored,
'Goals conceded': goals_conceded,
'Passes': passes,
'Shots': shots,
'Shots on target': shots_target
})

# fig = make_subplots(rows=3, cols=2, specs=[[{'type': 'polar'}]*2]*3,subplot_titles=df_teams['Team'])
fig_polar = make_subplots(rows=3, cols=2, specs=[[{'type': 'polar'}]*2]*3)

# Adding the Subplots for Arsena!

fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][5],
      r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
    ), 1, 1)

fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][5],
      r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
    ), 1, 2)

fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][5],
      r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
    ), 2, 1)

fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][5],
      r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=False
    ), 2, 2)

fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][5],
      r = df_teams.loc[5].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="orange"),showlegend=True
    ), 3, 1)

# Adding the Subplots for the champions

# Manchester United
fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][0],
      r = df_teams.loc[0].drop('Team').values.tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="red")
    ), 1, 1)

# Manchester City
fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][1],
      r = df_teams.loc[1].drop('Team').values.tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="deepskyblue")
    ), 1, 2)

# Chelsea
fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][2],
      r = df_teams.loc[2].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="navy")
    ), 2, 1)

# Leiscerter City
fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][3],
      r = df_teams.loc[3].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="cyan")
    ), 2, 2)

# Liverpool
fig_polar.add_trace(go.Scatterpolar(
      name = df_teams['Team'][4],
      r = df_teams.loc[4].drop('Team').values.flatten().tolist(),
      theta = df_teams.columns.tolist()[1:],line=dict(color="darkred")
    ), 3, 1)

# Adjusting the layout

layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000,

    xaxis= go.layout.XAxis(linecolor = 'black',
                          linewidth = 1,
                          mirror = True),

    yaxis= go.layout.YAxis(linecolor = 'black',
                          linewidth = 1,
                          mirror = True),

    margin=go.layout.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad = 4
    )
)

fig_polar.update_traces(fill='toself')

fig_polar.update_layout(
    autosize=False,
    width=1000,
    height=1000,
    yaxis=dict(
        title_text="Y-axis Title",
        ticktext=["Very long label", "long label", "3", "label"],
        tickvals=[1, 2, 3, 4],
        tickmode="array",
        titlefont=dict(size=30),
    )
)

fig_polar.update_yaxes(automargin=True)
# fig_polar.show()


# In[30]:


# Correlation between variables and training the model

# We will focus then on the variable points which determines at the end of each season if a Team is the champion or not

to_delete=["season","date","home_team","away_team",'home_goals','away_goals','points_home','points_away']
X = df.drop(to_delete, axis=1)
X['winner'] = df['winner'].replace({'H': 1, 'D': 0, 'A': -1})

scaler = StandardScaler()
X[X.columns]=scaler.fit_transform(X)

cor_matrix = X.corr()
plt.figure(figsize=(30,10))
sns.heatmap(cor_matrix[["winner"]], annot=True, cmap="RdBu_r")
plt.show();
# print(cor_matrix[["winner"]][((cor_matrix >= .1) | (cor_matrix <= -.1)) & (cor_matrix !=1.000)])

matrix_winner=sns.heatmap(cor_matrix[["winner"]], annot=True, cmap="RdBu_r")

variables=cor_matrix[["winner"]][((cor_matrix >= .1) | (cor_matrix <= -.1)) & (cor_matrix !=1.000)]

variables=variables.dropna()
variables=variables.index.values.tolist()
variables.append("winner")
variables


# In[31]:


# Training the model

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

data= df.drop(["season","date","points_home","points_away","winner",'home_team','away_team','home_goals','away_goals','W','L','D'], axis=1)
data=scaler.fit_transform(data)
# target = df['winner'].replace({'H': -1, 'D': 0, 'A': 1})
target = df['winner']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, shuffle=False)


# In[32]:


lr = LogisticRegression()
knn = KNeighborsClassifier()
clf=ensemble.RandomForestClassifier(n_jobs=-1,random_state=321)


lr.fit(X_train, y_train)
lr_score=lr.score(X_test, y_test)
# print("score of lr : {}".format(lr.score(X_test, y_test)))

knn.fit(X_train, y_train)
knn_score=knn.score(X_test, y_test)
# print("score of knn : {}".format(knn.score(X_test, y_test)))

clf.fit(X_train, y_train)
clf_score=clf.score(X_test, y_test)
# print("score of clf : {}".format(clf.score(X_test, y_test)))


# In[33]:


lr = LogisticRegression(random_state=123) 
parametres = {'C': np.linspace(0.05, 1, 20), 
              'penalty' : ['l1', 'l2', 'elasticnet', 'none'], 
              'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid_lr = GridSearchCV(estimator=lr, param_grid=parametres)

grid_lr.fit(X_train, y_train)
# print("Best Parameters : {}".format(grid_lr.best_params_))
# print("score of lr : {}".format(grid_lr.score(X_test, y_test)))

y_pred_lr = grid_lr.predict(X_test) 


# In[34]:

df_classification_report=pd.DataFrame(classification_report(y_test, y_pred_lr, output_dict=True))
# print(df_classification_report)


# In[35]:


cross_tab=pd.crosstab(y_test, y_pred_lr, rownames=['Real class'], colnames=['Predicted class'])

