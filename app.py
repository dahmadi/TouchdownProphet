import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import pickle
import sklearn
import os


st.set_page_config(page_title='Touchdown Prophecy', layout='wide')
st.image('photos/Main.png')
st.title('Touchdown Prophecy: NFL Game Predictor')

models = {'Logistic Regression': {'load' : pickle.load(open('LR_model.sav','rb'))},
         'XG Boost': {'load' : pickle.load(open('XGBoost_model.sav','rb'))},
         'MLP Regression': {'load' : pickle.load(open('MLP_model.sav','rb'))}
         }

sorted_models = ['Logistic Regression', 'XG Boost', 'MLP Regression']


path = ("Data/NFL_Predictions_df.csv")
com_data = pd.read_csv(path)

st.markdown("""
Welcome to Touchdown Prophecy your go-to platform for elevating your Super Bowl outcome predictions. Ascertaining the true value of the NFL betting scene poses a considerable challenge, fluctuating between a staggering 700 Billion to 1 Trillion. Additionally, the clandestine market further swells this figure by an estimated 500 Billion. If your objective is financial gain, rely on us to amplify your odds of success.
Thank you for choosing our application. We anticipate achieving victories together. May the Touchdown Prophecy be with you!
Disclaimer: This content is for informational purposes only and does not constitute betting advice.
""")

st.sidebar.header('Playoff Teams')


teams_dict = {'Buffalo Bills' : {'Abbrev':'BUF', 'Logo' : 'photos/Bills.png', 'Seed' : 2, 'Blurb' : "Buffalo Bills (11-6), first place, AFC East"},
              'Pittsburgh Steelers' : {'Abbrev':'PIT', 'Logo' : 'photos/Steelers.png', 'Seed' : 7, 'Blurb' : "Pittsburgh Steelers (10-7), third place, AFC North"}, 
              'Kansas City Chiefs' : {'Abbrev':'KAN', 'Logo' : 'photos/Chiefs.png', 'Seed' : 3, 'Blurb' : "Kansas City Chiefs (11-6), first place, AFC West"}, 
              'Baltimore Ravens' : {'Abbrev':'RAV', 'Logo' : 'photos/Ravens.png', 'Seed' : 1, 'Blurb' : "Baltimore Ravens (13-4), AFC Champions, first place, AFC North"},
              'Houston Texans' : {'Abbrev':'HTX', 'Logo' : 'photos/Texans.png', 'Seed' : 4, 'Blurb' : "Houston Texans (10-7), first place, AFC South"}, 
              'Los Angeles Rams' : {'Abbrev':'RAM', 'Logo' : 'photos/Rams.png', 'Seed' : 6, 'Blurb' : "Los Angeles Rams (10-7), second place, NFC West "},
              'Miami Dolphins' : {'Abbrev':'MIA', 'Logo' : 'photos/Dolphins.png', 'Seed' : 6, 'Blurb' : "Miami Dolphins (11-6), second place, AFC East"},
              'Tampa Bay Buccaneers' : {'Abbrev':'TAM', 'Logo' : 'photos/Buccaneers.png', 'Seed' : 4, 'Blurb' : "Tampa Bay Buccaneers (9-8), first place, NFC South"}, 
              'San Francisco 49ers' : {'Abbrev':'SFO', 'Logo' : 'photos/49ers.png', 'Seed' : 1, 'Blurb' : "San Francisco 49ers (12-5), NFC Champions, first place, NFC West"}, 
              'Detroit Lions' : {'Abbrev':'DET', 'Logo' : 'photos/Lions.png', 'Seed' : 3, 'Blurb' : "Detroit Lions (12-5), first place, NFC North"}, 
              'Dallas Cowboys' : {'Abbrev':'DAL', 'Logo' : 'photos/Cowboys.png', 'Seed' : 2, 'Blurb' : "Dallas Cowboys (12-5), first place, NFC East"}, 
              'Philadelphia Eagles' : {'Abbrev':'PHI', 'Logo' : 'photos/Eagles.png', 'Seed' : 5, 'Blurb' : "Philadelphia Eagles (11-6), second place, NFC East"}, 
              'Green Bay Packers' : {'Abbrev':'GNB', 'Logo' : 'photos/Packers.png', 'Seed' : 7,  'Blurb' : "Green Bay Packers (9-8), second place, NFC North"}, 
              'Cleveland Browns' : {'Abbrev':'CLE', 'Logo' : 'photos/Browns.png', 'Seed' : 5, 'Blurb' : "Cleveland Browns (11-6), second place, AFC North"}}


# Sidebar - Team selection

sorted_unique_team = ["Cleveland Browns", "Houston Texans", "Kansas City Chiefs","Miami Dolphins",
                      "Pittsburgh Steelers","Buffalo Bills","Green Bay Packers","Dallas Cowboys",
                      "Los Angeles Rams","Detroit Lions","Tampa Bay Buccaneers","Philadelphia Eagles",
                      "Baltimore Ravens","San Fransisco 49ers"]

selected_team = st.sidebar.selectbox("Teams", sorted_unique_team)

selected_model = st.sidebar.selectbox("Choose your Model", sorted_models)

# get index of every team's data
team_index = com_data['Team']

# Remove Opponent, Score, Result
model_data = com_data[['Team', 'Opponent', 'Points Scored', '1st Downs', 'Total Yards Gained', ' Passing Yards', ' Rushing Yards', 'Turnovers Lost',
                         '1st Downs Allowed', 'Total Yards Allowed', 'Passing Yards Allowed', 'Rushing Yards Allowed', 'Turnovers Gained', 'Home','Prediction_LR','Prediction_ADA']]

# standardise the data
from sklearn import preprocessing

sd_data = ['1st Downs', 'Total Yards Gained', ' Passing Yards', ' Rushing Yards', 'Turnovers Lost',
                         '1st Downs Allowed', 'Total Yards Allowed', 'Passing Yards Allowed', 'Rushing Yards Allowed', 'Turnovers Gained','Prediction_LR','Prediction_ADA']

model_data[sd_data] = preprocessing.scale(model_data[sd_data])

#get indexs of every teams
team_index = com_data['Team']

model_data = pd.get_dummies(model_data)

pd.options.display.max_rows = None

pd.options.display.max_columns = None

# Create playoff test dataset from season averages
@st.cache
def Score_Predictor(home_team, away_team):
    team1 = home_team
    team2 = away_team
    
    team1_data = model_data[com_data['Team'] == team1].drop('Points Scored', axis=1).reset_index(drop=True)
    team2_data = model_data[com_data['Team'] == team2].drop('Points Scored', axis=1).reset_index(drop=True)
    
    week_slice = slice(0,17)
    
    #1 Remove if no team names
    team1_test = pd.DataFrame(team1_data[week_slice].mean(axis=0)).T #select week to use as average
    #team1_test  #This was the line printing that extra dataframe onto the Dashboard
    opp_columns = team1_test.filter(like='Opponent').columns
    
    team1_test[opp_columns] = 0
    team1_test['Team_' + team2] = 1
    team1_test['Home'] = 1
    
    #2
    team2_test = pd.DataFrame(team2_data[week_slice].mean(axis=0)).T #select week to use as average
    opp_columns = team2_test.filter(like='Opponent').columns
    
    team2_test[opp_columns] = 0
    team2_test['Team_' + team1] = 1
    team2_test['Home'] = 1 # change to remove home field advantage
    
    # head to head matchup
    team1_test[['1st Downs Allowed','Total Yards Allowed','Passing Yards Allowed','Rushing Yards Allowed','Turnovers Gained']] = team2_test[['1st Downs','Total Yards Gained',' Passing Yards',' Rushing Yards','Turnovers Lost']]
    team2_test[['1st Downs Allowed','Total Yards Allowed','Passing Yards Allowed','Rushing Yards Allowed','Turnovers Gained']] = team1_test[['1st Downs','Total Yards Gained',' Passing Yards',' Rushing Yards','Turnovers Lost']]
    
    X_Playoff_test = pd.concat([team1_test, team2_test])
    X_Playoff_test.fillna(0, inplace = True) # added to address the NANs that was causing the error
    
    scores = models[selected_model]['load'].predict(X_Playoff_test)
   
    
    if scores[0] > scores[1]:
        winner = team1
    else:
        winner = team2
    
    print(winner, " are the WINNERS!!!")
    
    return scores, winner


# Web scraping of NFL player stats
@st.cache
def load_data(team): #year,
    url = "https://www.pro-football-reference.com/teams/" + teams_dict[selected_team]['Abbrev'].lower() + "/2023.htm"
    df = pd.read_html(url, header = 1)
    df = df[1]
    return df
teamstats = load_data(selected_team) #selected_year, 

st.header('Display 2023 Season Schedule, Results, and Statistics')
st.subheader('Current Team Selection: ' + selected_team)
st.image(teams_dict[selected_team]['Logo'], width = 500)
st.dataframe(teamstats)
st.write("Source: https://www.pro-football-reference.com/teams/" + f"{teams_dict[selected_team]['Abbrev'].lower()}" + "/2023.htm")

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href
st.markdown(filedownload(teamstats), unsafe_allow_html=True)


Buffalo_Bills = Image.open("photos/Bills.png")
Pittsburgh_Steelers = Image.open("photos/Steelers.png")
Kansas_City_Chiefs = Image.open("photos/Chiefs.png")
Baltimore_Ravens = Image.open("photos/Ravens.png")
Houston_Texans = Image.open("photos/Texans.png")
Los_Angeles_Rams = Image.open("photos/Rams.png")
Miami_Dolphins = Image.open("photos/Dolphins.png")
Tampa_Bay_Buccaneers = Image.open("photos/Buccaneers.png")
San_Francisco_49ers = Image.open("photos/49ers.png")
Detroit_Lions = Image.open("photos/Lions.png")
Dallas_Cowboys = Image.open("photos/Cowboys.png")
Philadelphia_Eagles = Image.open("photos/Eagles.png")
Green_Bay_Packers = Image.open("photos/Packers.png")
Cleveland_Browns = Image.open("photos/Browns.png")


st.subheader(" ")
st.header(f"{selected_team}' Playoff Picture")
st.write(teams_dict[selected_team]['Blurb'])
st.write('Source: https://www.sportingnews.com/us/nfl/news/nfl-playoff-picks-predictions-2024-afc-nfc-super-bowl-58/a845b0e848e89f626ca15e2b')

st.markdown('---')
st.title("Road to Super Bowl LVIII")


button1 = st.button("Run Touchdown Prophet")
col1, col2 = st.columns(2)
col1.title("Wild Card Round")

col1.subheader("AFC Game 1")
col1.image(teams_dict['Cleveland Browns']['Logo'], width = 200)
col1.image(teams_dict['Houston Texans']['Logo'], width = 200)
col1.subheader("AFC Game 2")
col1.image(teams_dict['Miami Dolphins']['Logo'], width = 200)
col1.image(teams_dict['Kansas City Chiefs']['Logo'], width = 200)
col1.subheader("AFC Game 3")
col1.image(teams_dict['Pittsburgh Steelers']['Logo'], width = 200)
col1.image(teams_dict['Buffalo Bills']['Logo'], width = 200)

col1.subheader("NFC Game 1")
col1.image(teams_dict['Green Bay Packers']['Logo'], width = 200)
col1.image(teams_dict['Dallas Cowboys']['Logo'], width = 200)
col1.subheader("NFC Game 2")
col1.image(teams_dict['Los Angeles Rams']['Logo'], width = 200)
col1.image(teams_dict['Detroit Lions']['Logo'], width = 200)
col1.subheader("NFC Game 3")
col1.image(teams_dict['Philadelphia Eagles']['Logo'], width = 200)
col1.image(teams_dict['Tampa Bay Buccaneers']['Logo'], width = 200)

col1.title("AFC Division Round")

col1.subheader("AFC Game 1")
col1.image(teams_dict['Houston Texans']['Logo'], width = 200)
col1.image(teams_dict['Baltimore Ravens']['Logo'], width = 200)
col1.subheader("AFC Game 2")
col1.image(teams_dict['Kansas City Chiefs']['Logo'], width = 200)
col1.image(teams_dict['Buffalo Bills']['Logo'], width = 200)

col1.title("NFC Division Round")
col1.subheader("NFC Game 1")
col1.image(teams_dict['Green Bay Packers']['Logo'], width = 200)
col1.image(teams_dict['San Francisco 49ers']['Logo'], width = 200)
col1.subheader("NFC Game 2")
col1.image(teams_dict['Tampa Bay Buccaneers']['Logo'], width = 200)
col1.image(teams_dict['Detroit Lions']['Logo'], width = 200)

col1.title("AFC Conference Championship")
col1.image(teams_dict['Kansas City Chiefs']['Logo'], width = 200)
col1.image(teams_dict['Baltimore Ravens']['Logo'], width = 200)

col1.title("NFC Conference Championship")
col1.image(teams_dict['San Francisco 49ers']['Logo'], width = 200)
col1.image(teams_dict['Detroit Lions']['Logo'], width = 200)

col1.subheader("AFC Bye Team ")
col1.image(teams_dict['Baltimore Ravens']['Logo'], width = 200)
col1.subheader("NFC Bye Team ")
col1.image(teams_dict['San Francisco 49ers']['Logo'], width = 200)
container = st.container()

afc_lowest = ''
afc_team1 = ''
afc_team2 = ''

nfc_lowest = ''
nfc_team1 = ''
nfc_team2 = ''

winner = ""

if button1:
    
    scores1, afc_winner1 = Score_Predictor('Cleveland Browns', 'Houston Texans')
    scores2, afc_winner2 = Score_Predictor('Miami Dolphins', 'Kansas City Chiefs')
    scores3, afc_winner3 = Score_Predictor('Pittsburgh Steelers', 'Buffalo Bills')
    scores4, nfc_winner1 = Score_Predictor('Green Bay Packers', 'Dallas Cowboys')
    scores5, nfc_winner2 = Score_Predictor('Los Angeles Rams', 'Detroit Lions')
    scores6, nfc_winner3 = Score_Predictor('Philadelphia Eagles', 'Tampa Bay Buccaneers')
    
    col2.title("Touchdown Prophecy")
    col2.subheader("AFC Game 1 Winner")
    col2.image(teams_dict[afc_winner1]['Logo'], width = 325, caption = f' Final Score: {scores1[0]: .0f} vs. {scores1[1]: .0f}')
    col2.subheader("AFC Game 2 Winner")
    col2.image(teams_dict[afc_winner2]['Logo'], width = 325, caption = f'Final Score: {scores2[0]: .0f} vs. {scores2[1]: .0f}')
    col2.subheader("AFC Game 3 Winner")
    col2.image(teams_dict[afc_winner3]['Logo'], width = 325, caption = f'Final Score: {scores3[0]: .0f} vs. {scores3[1]: .0f}')
    col2.subheader("NFC Game 1 Winner")
    col2.image(teams_dict[nfc_winner1]['Logo'], width = 325, caption = f'Final Score: {scores4[0]: .0f} vs. {scores4[1]: .0f}')
    col2.subheader("NFC Game 2 Winner")
    col2.image(teams_dict[nfc_winner2]['Logo'], width = 325, caption = f'Final Score: {scores5[0]: .0f} vs. {scores5[1]: .0f}')
    col2.subheader("NFC Game 3 Winner")
    col2.image(teams_dict[nfc_winner3]['Logo'], width = 325, caption = f'Final Score: {scores6[0]: .0f} vs. {scores6[1]: .0f}')
    col2.subheader("AFC Bye Team ")
    col2.image(teams_dict['Baltimore Ravens']['Logo'], width = 300)
    col2.subheader("NFC Bye Team ")
    col2.image(teams_dict['San Francisco 49ers']['Logo'], width = 300)
       
    if (teams_dict[afc_winner1]['Seed'] > teams_dict[afc_winner2]['Seed']) and (teams_dict[afc_winner1]['Seed'] > teams_dict[afc_winner3]['Seed']):
        afc_lowest = afc_winner1
        afc_team1 = afc_winner2
        afc_team2 = afc_winner3
    elif (teams_dict[afc_winner2]['Seed'] > teams_dict[afc_winner1]['Seed']) and (teams_dict[afc_winner2]['Seed'] > teams_dict[afc_winner3]['Seed']):
        afc_lowest = afc_winner2
        afc_team1 = afc_winner1
        afc_team2 = afc_winner3
    elif (teams_dict[afc_winner3]['Seed'] > teams_dict[afc_winner1]['Seed']) and (teams_dict[afc_winner3]['Seed'] > teams_dict[afc_winner2]['Seed']):
        afc_lowest = afc_winner3
        afc_team1 = afc_winner1
        afc_team2 = afc_winner2
        

    if (teams_dict[nfc_winner1]['Seed'] > teams_dict[nfc_winner2]['Seed']) and (teams_dict[nfc_winner1]['Seed'] > teams_dict[nfc_winner3]['Seed']):
        nfc_lowest = nfc_winner1
        nfc_team1 = nfc_winner2
        nfc_team2 = nfc_winner3
    elif (teams_dict[nfc_winner2]['Seed'] > teams_dict[nfc_winner1]['Seed']) and (teams_dict[nfc_winner2]['Seed'] > teams_dict[nfc_winner3]['Seed']):
        nfc_lowest = nfc_winner2
        nfc_team1 = nfc_winner1
        nfc_team2 = nfc_winner3
    elif (teams_dict[nfc_winner3]['Seed'] > teams_dict[nfc_winner1]['Seed']) and (teams_dict[nfc_winner3]['Seed'] > teams_dict[nfc_winner2]['Seed']):
        nfc_lowest = nfc_winner3
        nfc_team1 = nfc_winner1
        nfc_team2 = nfc_winner2

    
    st.markdown('---')
    col1, col2 = st.columns(2)
    col1.title("Divisional Round")
    col1.subheader("AFC Divisional Game 1")
    col1.image(teams_dict['Baltimore Ravens']['Logo'], width = 200)
    col1.image(teams_dict[afc_lowest]['Logo'], width = 200)

    col1.subheader("AFC Divisional Game 2")
    col1.image(teams_dict[afc_team1]['Logo'], width = 200)
    col1.image(teams_dict[afc_team2]['Logo'], width = 200)

    col1.subheader("NFC Divisional Game 1")
    col1.image(teams_dict['San Francisco 49ers']['Logo'], width = 200)
    col1.image(teams_dict[nfc_lowest]['Logo'], width = 200)

    col1.subheader("NFC Divisional Game 2")
    col1.image(teams_dict[nfc_team1]['Logo'], width = 200)
    col1.image(teams_dict[nfc_team2]['Logo'], width = 200)
    
    scores7, winner7 = Score_Predictor('Baltimore Ravens', afc_lowest)
    scores8, winner8 = Score_Predictor(afc_team1, afc_team2)
    scores9, winner9 = Score_Predictor('San Francisco 49ers', nfc_lowest)
    scores10, winner10 = Score_Predictor(nfc_team1, nfc_team2)
   
    col2.title("Touchdown Prophet")  
    col2.subheader("AFC Divisional Game 1 Winner")
    col2.image(teams_dict[winner7]['Logo'], width = 350, caption = f'Final Score: {scores7[0]: .0f} vs. {scores7[1]: .0f}')

    col2.subheader("AFC Divisional Game 2 Winner")
    col2.image(teams_dict[winner8]['Logo'], width = 350, caption = f'Final Score: {scores8[0]: .0f} vs. {scores8[1]: .0f}')

    col2.subheader("NFC Divisional Game 1 Winner")
    col2.image(teams_dict[winner9]['Logo'], width = 350, caption = f'Final Score: {scores9[0]: .0f} vs. {scores9[1]: .0f}')

    col2.subheader("NFC Divisional Game 2 Winner")
    col2.image(teams_dict[winner10]['Logo'], width = 350, caption = f'Final Score: {scores10[0]: .0f} vs. {scores10[1]: .0f}')
        
    st.markdown('---')
    col1, col2 = st.columns(2)
    col1.title("Conference Championships")
    col1.subheader("AFC Final Game")
    col1.image(teams_dict[winner7]['Logo'], width = 200)
    col1.image(teams_dict[winner8]['Logo'], width = 200)

    col1.subheader("NFC Final Game")
    col1.image(teams_dict[winner9]['Logo'], width = 200)
    col1.image(teams_dict[winner10]['Logo'], width = 200)

    scores11, winner11 = Score_Predictor(winner7, winner8)
    scores12, winner12 = Score_Predictor(winner9, winner10)

    col2.title("Touchdown Prophet")
    col2.subheader("AFC Final Winner")
    col2.image(teams_dict[winner11]['Logo'], width = 400, caption = f'Final Score: {scores11[0]: .0f} vs. {scores11[1]: .0f}')
            
    col2.subheader("NFC Final Winner")
    col2.image(teams_dict[winner12]['Logo'], width = 400, caption = f'Final Score: {scores12[0]: .0f} vs. {scores12[1]: .0f}')
               
    st.markdown('---')
    col1, col2 = st.columns(2)
    col1.title("Super Bowl")
    col1.image(teams_dict[winner11]['Logo'], width = 200)
    col1.image(teams_dict[winner12]['Logo'], width = 200)

    col2.title("Super Bowl Winner")
    scores13, winner13 = Score_Predictor(winner11, winner12)
          
    col2.image(teams_dict[winner13]['Logo'], width = 600, caption = f'Final Score: {scores13[0]: .0f} vs. {scores13[1]: .0f}')

st.sidebar.header('Create Your Own Game!')   
selected_team1 = st.sidebar.selectbox("Home Teams", sorted_unique_team)
selected_team2 = st.sidebar.selectbox("Away Teams", sorted_unique_team, index=1)

st.markdown('---')
st.markdown('---')
col1, col2 = st.columns(2)
col1.subheader("Single Game Matchup")
col1.image(teams_dict[selected_team1]['Logo'], width = 200)
col1.text("  ")
col1.image(teams_dict[selected_team2]['Logo'], width = 200)

button1 = st.sidebar.button("Show me the result using Touchdown Prophet")
if button1:  
    scores, winner = Score_Predictor(selected_team1, selected_team2)
    
    col2.subheader("Single Match Playoff")
    col2.image(teams_dict[winner]['Logo'], width = 400, caption = f'Final Score: {scores[0]: .0f} vs. {scores[1]: .0f}')
    
