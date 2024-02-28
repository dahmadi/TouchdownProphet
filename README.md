# TouchdownProphet

Welcome to Touchdown Prophecy, your go-to platform for elevating your Super Bowl outcome predictions. Ascertaining the true value of the NFL betting scene poses a considerable challenge, fluctuating between a staggering 700 Billion to 1 Trillion. Additionally, the clandestine market further swells this figure by an estimated 500 Billion. If your objective is financial gain, rely on us to amplify your odds of success.
Thank you for choosing our application. We anticipate achieving victories together. May the Touchdown Prophecy be with you!

**Disclaimer: This content is for informational purposes only and does not constitute betting advice.**

### Contributors

- James Davidson  
- Donya Ahmadi  
- Kobe Buncu  

## Requirements

This project utilizes python 3.10 Streamlit and scikit-learn.

A [conda](https://docs.conda.io/en/latest/) environment with liabraries listed below and [Jupyter Notebook/Lab](https://jupyter.org/) are required to run the code.

The following library was used:

1. [Scikit Learn](https://scikit-learn.org/stable/index.html) - Scikit Learn or Sklearn is one of the most used Python libraries for Data Science, along with others like Numpy and Pandas.

2. [Streamlit](https://streamlit.io/) - Streamlit turns data scripts into shareable web apps in minutes.


Install the following librarie(s) in your terminal...

    pip install -U scikit-learn
    pip install streamlit
 
---

## Data

The CSV files used in our codes were mainly created from https://pro-football-reference.com

![csv_file_list](photos/df_list.png)

---

## Process & Visualizations

The first step in our goal of predicting Super Bowl winners was to collect historical data from NFL teams. We opted to collect 10 years of Offense and Defense team statistics dating back to 2014. The following visualizations illustrate the original dataframes that were read from the csv files, collected from pro football reference. Additionally, we also downloaded data from the past 10 years of Super Bowl winners.    

Offense:

![offense_list](photos/offense_datasets.png)

Defense:

![defense_list](photos/defense_datasets.png)

Super Bowl:

![sb_list](photos/sb_dataset.png)

After collecting all the necessary data, we proceeded to clean the various dataframes and then combined the dataframes by 'Year' and 'Team' until we were left with a single dataframe. Once the final dataframe was generated we proceeded to standardize the 10 years of Team data, via `StandardScaler`, and issue Offensive and Defensive ranks, by way of y-hat predicted values using, both `LogisticRegression` and `AdaBoostClassifier` models. We then saved each model into separate csv files, which will be required in building our machine-learning models.

Completed Dataframe:

![final_df](photos/training_set_df.png)

LR Prediction Values:

![lr_pred](photos/lr_predictions.png)

ADA Prediction Values:

![ada_pred](photos/ada_predictions.png)