# Importing database and libraries

import warnings

warnings.filterwarnings('ignore')

# get_ipython().run_line_magic('matplotlib', 'inline')

import streamlit as sl


# Importing python files
import sys
import os
import io

sys.path.append(os.path.abspath("C:/Users/Sebastian Arce/Documents/Data Science/streamlit_app/Project_Arsenal"))
from Project_Arsenal.database_variables import *
from Project_Arsenal.Arsenal_Project import *


# Database

def inject_CSS_table(xzy):
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    sl.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    # Display a static table
    sl.table(xzy)


def inject_CSS_dataframe(xyz):
    # CSS to inject contained in a string
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    sl.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    # Display an interactive table
    sl.dataframe(xyz)


df_deleted = deleted
# .css-yyj0jg.edgvbvh3
# css-h5rgaw egzxvld1 Footer

sl.markdown("""
<style>

.css-9s5bis.edgvbvh3
{
    visibility:hidden;
}
.css-h5rgaw.egzxvld1
{
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)

# Creating a sidebar
opt = sl.sidebar.selectbox('# Content', options=['Homepage', 'Introduction', 'Objective of the Project',
                                             'Methodology', 'Analysis and Results','Interactive graph',
                                             'Conclusions'])

# Containers
homepage = sl.container()
introduction = sl.container()
objective = sl.container()
methodology = sl.container()
analysis_results = sl.container()
interactive_graph = sl.container()
conclusions = sl.container()

if opt == 'Homepage' or opt is None:
    with homepage:
        sl.image('webpage_image.jpeg', width=400)
        sl.title("Arsenal FC: The decline of one of the greats")
        sl.subheader('By Izat, Paul and Sebastian')
        sl.write('This project is based on the performance of Arsenal FC over the last decade. '
                 'Specifically, this project analyze the data from the Premier League matches from 2010 '
                 'to tell how has been the performance of Arsenal.')

if opt == 'Introduction':
    with introduction:
        sl.title("Introduction")
        with sl.expander('*Description of Arsenal FC*'):
            
            sl.write("Arsenal Football Club is an English professional soccer club based in London and plays in the "
                 "Premier League, the top tier of English soccer."
                 "The club has won 13 league titles, a record 14 FA Cups, two League Cups, 16 FA Community Shields, "
                 "one European Cup Winners' Cup and one Inter-Cities Fairs Cup."
                 "In terms of trophies won, they are the third most successful club in English soccer (Wikipedia n.d.)."
                 "However, despite being one of the most successful teams, one of the richest, always counting with "
                 "highly recognized players and having one of the largest fans in England,"
                 "its football performance has been decreasing over the years and this is evidenced by the fact that "
                 "the last time Arsenal was champion of the Premier League was in the 2003/2004 season."
                 )
            
        with sl.expander('*Description of the database*'):
            sl.write("The database to be used has been obtained from the Kaggle website. This database has important "
                 "statistics of Premier League matches from 2010 to mid-2021 (Kaggle 2021). Similarly, this data is "
                 "available for free on the English Premier League website and allows tracking of the last few "
                 "seasons. The original and updated archive is available on the official Premier League website.")

if opt == 'Objective of the Project':
    with homepage:
        sl.image('objectives.jpg', width=300)
        sl.title("Objective of the Project")
        with sl.expander('*Main Objective*'):
            sl.write("Develop a data storytelling in order to explain how the level of the football club Arsenal FC has "
                 "deteriorated during the last decade.")
        
        with sl.expander('Specific objectives'):
            sl.write("•	Explore and clean the database to obtain key parameters that help determine the performance of a "
                 "football team.")
            sl.write("\n"
                 "•	Creation of a variable indicating the number of points obtained at the end of a match")
            sl.write("\n"
                 "•	Creation of the final rankings of each season.")
            sl.write("\n"
                 "•	Compare the performance of Arsenal against the best teams.")
            sl.write("\n"
                 "•	Compare the performance of Arsenal against the champions of the premier league.")
            sl.write("\n"
                 "•	Develop a prediction model to determine the winner of the match.")

if opt == 'Methodology':
    with methodology:
        sl.title("Methodology")
        with sl.expander('*Importing libraries and database*'):
            code_modules = ("""
                import warnings
                from IPython import get_ipython
                warnings.filterwarnings('ignore')
                
                import json
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                import matplotlib.patches as mpatches
                get_ipython().run_line_magic('matplotlib', 'inline')
                
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
                
                import streamlit as sl
                """)
            sl.write("The first step is to import the necessary libraries to be able to explore, analyze and visualize "
                 "the information of the database. For this purpose, the libraries to be used are as follows")
            sl.code(code_modules)
            sl.write("Once the libraries have been imported, the next step is to import the database. The database is a "
                 "csv file which can be imported using the function 'read_csv' of the pandas’ module. This database "
                 "has a size of 4070,114. In other words, it has a total of **4070 entries(matches)** and also has "
                 "**114"
                 "columns (variables)**.")
            sl.dataframe(original_df.head())
        with sl.expander('*Removing unnecessary information*'):
            sl.write(r"The database has a large number of variables, which contain important information on every game "
                 r"played since 2010. However, it is necessary to reduce this number in order to perform an adequate "
                 r"analysis of the teams' performance.")
            sl.write("\n"
                 r"The column 'Unnamed: 0' is identical to the index. Since the index is already integrated in the "
                 r"table, there is no need to store it in a separate column. The column 'link_match' represents links "
                 r"that directs you to the match information. There is no need to keep this column since most of the "
                 r"data is already stored in other columns.")
            sl.write("\n"
                 r"To further study the performance of a team we need information related to offensive and defensive "
                 r"indicators. The database contains too much information on these aspects. This information is "
                 r"contained in different columns that represent halftime and fulltime match results data. For this "
                 r"analysis, the final score of the match is required, which helps to determine who was the winner.")
            sl.write("\n"
                 r"Therefore, the following columns will not be considered are shown as follows")
            
            inject_CSS_table(deleted)
            sl.write("Additionally, 5 new columns are added containing the home team goals, away team goals, home team "
                 "points, away team points and the winner of the match. To determine the points obtained, "
                 "it is enough to simply compare the number of goals scored by each team. The team that scored the "
                 "most goals at the end of the match gets 3 points and is the winner. Both teams get 1 point if they "
                 "scored the same number of goals. The team with fewer goals scored gets 0 points and is the loser.")

        with sl.expander('*Identifying and/or deleting duplicate values and NaN management*'):
            sl.write("Datasets that contain duplicates and missing values may contaminate the training data with the test "
                 "data or vice versa as well as lead to wrong interpretation of the different variables. A good way "
                 "to get a quick feel for the data is to take a look at the information of the DataFrame. As shown on "
                 "the following, the Dataframe has a total of 4070 entries, 35 columns and non-null values were found.")
            df_info = pd.DataFrame({"name": df.columns, "non-nulls": len(df) - df.isnull().sum().values,
                                "nulls": df.isnull().sum().values, "type": df.dtypes.values})
            inject_CSS_table(df_info)
            
            sl.write("Entries with missing values will lead models to misunderstand features, and outliers will undermine "
                 "the training process – leading the model to “learn” patterns that do not exist in reality. To "
                 "identify duplicate values the 'duplicated' function is used in combination with the 'sum' function. "
                 "If the sum is greater than 0 it means that duplicates have been found. For this specific database "
                 "no duplicate values were found.")

        with sl.expander('*Adjusting the data*'):
            sl.write("For the elaboration of the graphs a new dataframe will be created. This dataframe will contain the "
                 "statistics per season from 2010 to 2020. The 2021 season will not be considered because the "
                 "available data only has information up to the middle of the season. This is done in order to have "
                 "the information more summarized and easier to interpret visually.")
            code_season_total = ("""def season_total(df, home_field, away_field,name_1,name_2):
                                  home_grouped = df.groupby(['season','home_team'], as_index = False).agg({home_field:'sum'})
                                  away_grouped = df.groupby(['season','away_team'], as_index = False).agg({away_field:'sum'})
                                  grouped = home_grouped.merge(away_grouped,how="inner",on="season")
                                  grouped=grouped.rename(columns={home_field: name_1, away_field: name_2})
                                  return grouped
                            """)
            code_season_avg = ("""def season_avg(df, home_field, away_field,name_1,name_2):
                                home_grouped = df.groupby(['season','home_team'], as_index = False).agg({home_field:'mean'})
                                away_grouped = df.groupby(['season','away_team'], as_index = False).agg({away_field:'mean'})
                                grouped = home_grouped.merge(away_grouped,how="inner",on="season")
                                grouped=grouped.rename(columns={home_field: name_1, away_field: name_2})
                                return grouped
                        """)
            sl.code(code_season_total)
            sl.code(code_season_avg)
            sl.write("As shown in the figures above, two functions were created to obtain the total statistics per season "
                 "as well as the average per season. These functions receive as input the database, the columns to be "
                 "considered and the name to be assigned to them.")
            sl.write("\n"
                 "For example, to calculate the total points per season we will use the following inputs df, "
                 "'points_home','points_away','Points(home)','Points(away)'. This will create a new dataframe "
                 "containing the total home points and away points per season for each team. Analogously, "
                 "the total and average statistics will be obtained for the following information: goals scored, "
                 "goals conceded, passes, possession, shots and shots on target.")
            sl.write("\n"
                 "As shown in the figure below, the total and average statistics dataframes per season will be merged "
                 "into one big dataframe called 'performance_per_season' with the help of the 'merging' function. This "
                 "database will be the basis for the creation of graphs.")
            code_performance_season = ("""performance_per_season=points_per_season.merge(goals_scored_per_season,how="inner",
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
                                               """)
            sl.code(code_performance_season)
            
            sl.write("Once the dataframes are merged, the rows in which the team name is the same in the columns 'team "
                 "home' and 'team away' will be deleted. This is because when the merging is done, rows are created "
                 "between the matches of each team against the others. This information is not relevant because the "
                 "objective is to determine the statistics of each team per season and not how they performed against "
                 "other teams.")
            sl.write("\n"
                 "New columns were created to summarize the performance information of each team. To "
                 "obtain the totals, simply add the home and away columns for each aspect of the match. For example, "
                 "for the total number of points, the Points(home) and Points(away) columns will be added together. "
                 "Similarly, this will be done for goals scored, goals conceded, passes, shots and shots on target.")
            sl.write("\n"
                 "For the average, the home and away columns are added together, and the result is divided by 2. "
                 "Finally, the result is rounded and expressed with two decimal places. Additionally, a new column "
                 "called 'Table_Standing' will be created. This column will show the position occupied by each team "
                 "at the end of the season. In order to determine who is the season champion, the data is drawn "
                 "according to the number of points. In case two or more teams have the same number of points, "
                 "they are drawn according to the goal difference.")
            sl.write("\n"
                 "In order to evaluate a team's performance,"
                 "it is necessary to compare its statistics against the best teams of each season. For this report, "
                 "the best teams are defined as the teams that placed in the top four. The top four teams are the "
                 "ones that qualify for the Champions League, the most important club championship in Europe. The "
                 "best teams are shown in the figure below.")
            sl.markdown("""| Team | City |
| ----------- | ----------- |
| Arsenal | London |
| Chelsea | London |
| Leicester City | Leicester |
| Liverpool | Liverpool |
| Manchester City | Manchester |
| Manchester United | Manchester |
| Tottenham Hotspur | London |
                """)
        
        with sl.expander("*Predictive modeling*"):
            sl.write(
                "Predictive modeling is a mathematical process used to predict future outcomes by analyzing patterns in a given set of input data. It is a type of data analysis that uses current and historical data to predict activities, behaviors and trends (TechTarget n.d.).")
            sl.write("\n"
                 "For this research report, a predictive model will be developed to determine the winner of a game using information on the offensive and defensive aspects of each team. The problem in question is a classification problem. The developed model must be able to determine if a team is a winner, loser, or was a draw.")
            sl.write("\n"
                 "For this, three machine learning algorithms will be implemented in Python using Scikit-learn. The machines learning algorithms to be trained are LogisticRegression, KNeighborsClassifier and RandomForestClassifier. Then, the objective is to identify the machine learning algorithm that best fits the problem by comparing their performances and selecting the one with the best score.")
            sl.write("\n"
                 "The first step is to create a standardizer because the variables to be considered have different scales. In parallel, a correlation matrix is created to determine the relationship between the different variables. ‘season’,’date’,’home_team’,’away_team’,'home_goals','away_goals','points_home' and 'points_away' will not be considered in the correlation matrix. The information contained in these columns directly determines the winner or loser of a match because they have the final result of the match.")
            sl.write("\n"
                 "Then the ‘data’ and ‘target’ variables are created, which will be used to train the model. The ‘data’ variable contains all the variables discarding those mentioned above and also discarding the winner column. The ‘target’ variable will only consider the information of the winner column.")
            sl.write("\n"
                 "Finally, the model is trained so that it can be used with the algorithms. The variables ‘lr’,’knn’ and ‘clf’ are created, which represent respectively the algorithms LogisticRegression, KNeighborsClassifier and RandomForestClassifier. Now, the score of each algorithm is evaluated and the best one is chosen. This algorithm is optimized to improve its performance in order to obtain better classification results.")

if opt == 'Analysis and Results':
    with analysis_results:
        sl.header('Analysis and Results')
        with sl.expander("*Performance of the English Premier League’s Teams*"):
            sl.write("The performance of the best and worst teams is analyzed according to points obtained, goals scored ("
                 "offensive aspect) and goals conceded (defensive aspect).")
            sl.write("\n"
                 "As shown in the figure below, the team with the best performance is Manchester City."
                 "This team has managed to"
                 "stay in the top positions since 2010. Arsenal on the other hand has been in decline. From the 10/11 to "
                 "15/16 season they managed to stay in the top 4. However, it began to occupy lower positions, "
                 "reaching the eighth position in the table in the 19/20 season. Leicester was the surprise team winning "
                 "the premier league in 14/15. However, apart from this achievement, the team has had a regular "
                 "performance.")
            table_standing = graph_seaborn(filtered_data, 'Season_number', 'Table_standing', 'Table standing by season')
            
            sl.pyplot(table_standing)
            
            sl.write("As shown in the figure below, Manchester city is the team with the best offensive performance (goals "
                 "scored). The rest of the teams show ups and downs. Arsenal's performance in this aspect is a little "
                 "above (excluding Manchester city). However, from 16/17 season onwards, the number of goals scored by "
                 "Arsenal started to decline.")
            goals_scored = graph_seaborn(filtered_data, 'Season_number', 'Total_goals_scored', 'Goal scored by season')
            
            sl.pyplot(goals_scored)
            
            sl.write(
            "Manchester City is the team with the best defensive performance (goals conceded). Arsenal's performance "
            "has worsened in this aspect. As shown in the figure below,from the 15/16 season onwards, Arsenal's number"
            "of goals conceded started to increase. It can be inferred that Arsenal has been scoring fewer and fewer "
            "goals and conceding more and more goals. This is a clear indication of the decline in their "
            "performance. As a team begins to get more conceded more goals than scored, it means that they are not "
            "winning enough games to challenge for top spot in the table.")
            goals_conceded = graph_seaborn(filtered_data, 'Season_number', 'Total_goals_conceded',
                                       'Goal conceded by season')
            sl.pyplot(goals_conceded)
            sl.write(
            "An important aspect to evaluate the performance of a football team is the number of points obtained at "
            "the end of each season. Below are shows the English league champions since 2010. It is important to note "
            "that according to the data provided, for a team to be crowned champion, it must obtain more than 80 "
            "points at the end of the season and the number of goals scored must be greater than the number of goals "
            "conceded (around 50%).")
            sl.dataframe(champions_data[champions_data['Table_standing'] == 1])
            sl.write("From 10/11 onwards, arsenal has not scored more than 80 points per season. As shown in the figure "
                 "below, Arsenal's best season was in 13/14, in which the team scored a total of 79 points. "
                 "Nevertheless, this was not enough, and Arsenal ranked fourth in the table. The champion of this "
                 "season was Manchester City and they obtained 86 points.")
            arsenal_points = graph_seaborn(filtered_data[filtered_data['Team'] == 'Arsenal'],
                                       'Season_number', 'Total_goals_conceded',
                                       'Points obtained over the last decade by Arsenal')
            
            sl.pyplot(arsenal_points)
            
            sl.write("To compare more concretely the performance of the best teams and the arsenal, a polar chart was "
                 "made. It is used to demonstrate data in two-dimensional for two or more data series. The axes start "
                 "on the same point in a radar chart. This chart is used to compare more than one or two variables. "
                 "For this report, this chart is composed of variables goals scored, goals conceded, passes, points, "
                 "shots and shots on target. The variables were standardized on a scale of 0 to 3 so that they can be "
                 "better represented visually.")
            sl.write("\n"
                 "Arsenal's offensive performance is very similar to the performance of the last Premier League "
                 "champions. The number of goals scored, and points scored is very similar between Arsenal, "
                 "Liverpool, Manchester United and Chelsea.  Manchester city is the team that clearly has the best "
                 "score in each variable.")
            sl.write("\n"
                 "On the other hand, the defensive aspect is definitive to understand the performance of the teams. "
                 "As shown in the figure below, despite having similar statistics on the offensive performance, "
                 "Arsenal concedes more goals than the champions. This translates to arsenal's defensive performance "
                 "being below that of other teams. The exception is Leicester City whose performance is below what "
                 "would normally be expected of a champion.")
            
            sl.plotly_chart(fig_polar, use_container_width=True)
            
        with sl.expander('*Predictive modeling*'):
            sl.write("The first step was to develop the correlation matrix to determine the relationship between the "
                 "variables. As shown in the figure below, the variables most closely related to the winner of a "
                 "match are: 'home_clearances','home_passes','home_possession','home_red_cards','home_shots',"
                 "'home_shots_on_target','home_touches','home_yellow_cards','away_clearances','away_passes',"
                 "'away_possession','away_shots','away_shots_on_target','away_touches','winner'")
            
            fig_heat, ax_heat = plt.subplots(figsize=(12, 12))
            sns.heatmap(cor_matrix[["winner"]], annot=True, cmap="RdBu_r")
            
            sl.pyplot(fig_heat)
            
            sl.write("The model is trained after standardizing the data and creating the data and target variables."
                 "Then, the scores of each algorithm are evaluated to determine the best one. "
                 "As shown in the table below, the logistic regression algorithm has a score of 0.66 which is higher "
                 "compared to the other algorithms. Therefore, this algorithm is preferred because it can better "
                 "classify whether a team is a winner.")
            
            df_algorithms = pd.DataFrame({"Machine learning algorithm": ["LogisticRegression", "KNeighborsClassifier",
                                                                     "RandomForestClassifier"],
                                      "Score": [lr_score, knn_score, clf_score]})
            
            inject_CSS_table(df_algorithms)
                            
            sl.write("The parameters used in the initial analysis were the default parameters provided by Python. To "
                 "obtain better results, the model will be optimized using different parameters. Below are shown the "
                 "parameters that were used to optimize the model. Using the python GridSearchCV module it was "
                 "determined that the best parameters are: {'C': 0.05, 'penalty': 'none', 'solver': 'saga'}")
            
            code_parameters = ("""lr = LogisticRegression(random_state=123)
                            parametres = {'C': np.linspace(0.05, 1, 20),
                                          'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
                                          'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
                            grid_lr = GridSearchCV(estimator=lr, param_grid=parametres)
                            """)
            sl.code(code_parameters)
            
            sl.write("Despite optimizing the model, the score did not improve considerably. However, it is a good score "
                 "and it can be concluded that the algorithm performs an adequate classification of which team will "
                 "be the winner. Below are shown the cross matrix in which the classification performed by the "
                 "algorithm can be observed. The letter A represents that the visiting team will be the winner, "
                 "the letter D indicates that it will be a tie and the letter H indicates that the home team will be "
                 "the winner.")
            
            sl.table(df_classification_report)
            fig_crosstab = plt.subplots(figsize=(12, 12))
            stacked_data = cross_tab.stack(level=1)
            stacked_data.plot.bar()
            sl.pyplot(fig_crosstab)

if opt=='Interactive graph':
    with interactive_graph:
        sl.header('Interactive graph')
        sl.write('This section contains an interactive graph created with the Bokeh module. This graph allows to '
                 'evaluate each aspect (or variable) of the database by season. In the toolbar below, simply select '
                 'the variable to be analyzed and on the right side choose the teams to be compared.')

        variables_performance=list(performance_per_season.columns.values)
        variables_performance.remove('Season')
        variables_performance.remove('Team')
        variables_performance.remove('Season_number')

        select=sl.selectbox('Choose a variable',options=variables_performance)
        print(select)
        graph_bokeh = graph_lines(data=performance_per_season, variable=select)
        sl.bokeh_chart(graph_bokeh)



if opt == "Conclusions":
    with conclusions:
        sl.header("Conclusions")
        sl.write("Arsenal has been one of the best teams in the Premier league and has been at the top of the table. "
                 "However, over the years their performance has been declining and they have not been able to become"
                 "champions.The database of the Premier league matches from 2010 to 2021 has been adjusted to obtain"
                 "the relevant information to evaluate a team's performance. New variables were created to determine"
                 "the winning team of each match as well as the points obtained.")

        sl.write("\n"
                 "Arsenal's performance has been compared against the top teams "
                 "(defined as the teams in the top 4 of the table) as well as the champions. "
                 "Arsenal have been in the top four during the 10/11 to 15/16 seasons. "
                 "However, from this season onwards they started to occupy lower positions resulting in their "
                 "non-participation in the champions league.")

        sl.write("\n"
                 "Offensive and defensive aspects have been analyzed to evaluate arsenal's performance.On the one hand,"
                 "Arsenal has similar statistics in the offensive aspect compared to the best teams and champions.Their"
                 "goals scored ratio is similar. However, their offensive aspect has been declining causing them to "
                 "concede more goals than they score. This decline is due to their defensive performance. On average, "
                 "champion teams concede 33 goals while scoring 87 goals. Arsenal on average concede 44 goals while "
                 "scoring 70 goals.")

        sl.write("\n"
                 "Another important aspect is the number of points obtained by the arsenal.To be titled champion of the"
                 "Premier League,a team must obtain at least 80 points. Arsenal's best season was 13/14 with 78 points."
                 "Its worst season was 19/20 in which it obtained 56 points.")

        sl.write("\n"
                 "Class 'A' represents the visiting team as the winner. The model used 348 data points, of which it "
                 "predicted 351. In other words, it correctly assigned the class about 72% of the time. Class 'H' "
                 "represents the home team as the winner. The model used 444 data points, of which it predicted 385, "
                 "i.e. it correctly assigned the class about 86% of the time. Class 'D' representing tie was the class "
                 "with the least accuracy. The model used 226 data, of which it predicted 35. It correctly assigned the"
                 "class about 15% of the time.")

# Brief description
#
# sl.write('This project is based on the performance of Arsenal FC over the last decade. '
#           'Specifically, this project analyze the data from the Premier League matches over the last decade '
#           'to tell how has been the performance of Arsenal.')

# sl.latex(r"\begin{pmatrix}\\a&b\\c&d\end{pmatrix}")

# sl.table(df.head(5)) #shows as a static table

# sl.dataframe(df.head(5))  # shows interactive table
#
# x = np.linspace(0, 10, 100)

# fig, ax = plt.subplots(figsize=(12,12))
# sns.lineplot(data=filtered_data, x='Season_number',y='Table_standing',hue='Team')
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
# sl.pyplot(fig)

# seaborn_grpah=graph_seaborn(filtered_data,'Season_number','Table_standing')
#
# sl.pyplot(seaborn_grpah)
#
# points_per=graph_lines(data=performance_per_season,variable='Total_points')
