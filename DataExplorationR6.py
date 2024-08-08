import pandas as pd
import mlflow
import pandas.core.series
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn import tree

# In Absprache mit dem Dozenten werden die Daten nicht mit abgegeben, da diese zu groß sind
# Kaggle-Link: https://www.kaggle.com/datasets/maxcobra/rainbow-six-siege-s5-ranked-dataset

'''
all_files = ['R6Archive/datadump_s5-000.csv', 'R6Archive/datadump_s5-001.csv', 'R6Archive/datadump_s5-002.csv',
            'R6Archive/datadump_s5-003.csv','R6Archive/datadump_s5-004.csv','R6Archive/datadump_s5-005.csv',
            'R6Archive/datadump_s5-006.csv', 'R6Archive/datadump_s5-007.csv', 'R6Archive/datadump_s5-008.csv',
            'R6Archive/datadump_s5-009.csv','R6Archive/datadump_s5-010.csv','R6Archive/datadump_s5-011.csv',
            'R6Archive/datadump_s5-012.csv','R6Archive/datadump_s5-013.csv','R6Archive/datadump_s5-014.csv',
            'R6Archive/datadump_s5-015.csv',
            'R6Archive/datadump_s5-016.csv', 'R6Archive/datadump_s5-017.csv','R6Archive/datadump_s5-018.csv',
             'R6Archive/datadump_s5-019.csv', 'R6Archive/datadump_s5-020.csv', 'R6Archive/datadump_s5-021.csv']
'''


def main():
    # selected_rows_columns = preprocessing1_selection(all_files)
    # Preprocessing1Result.csv

    # Works from here on.
    # Data is being read from file because original data is too large to hand in.
    filename = "Preprocessing1Result.csv"
    selected_rows_columns = pd.read_csv(filename, index_col=None, header=0)

    training_data_raw = preprocessing2_logical_df(selected_rows_columns)
    # Preprocessing2LogicalResult.csv
    training_data_encoded = one_hot_encode(training_data_raw)
    # mainResultEncoded.csv

    features_train, labels_train, features_validation, labels_validation, features_test, labels_test = split_train_val_test(training_data_encoded)

    '''
    Beginning of hyperparameter-tuning
    Parameters:
        maxDepth, criterion 
    '''

    #current_best_depth, current_best_criterion = hyperparam_tuning(features_train, labels_train, features_validation, labels_validation)

    '''
    --- Hyperparameter tuning results ---
    Best Score: 0.5325930495578861
    Best Depth: 12
    Best Criterion: entropy
    '''

    # Train models with hyperparameters around optimum and visualize with mlflow
    current_best_criterion = 'entropy'
    current_best_depth = 12
    criterion_list = ['gini', 'entropy', 'log_loss']

    mlflow.sklearn.autolog()
    for criterion in criterion_list:
        for maxDepth in range(1, 20):
            model = tree.DecisionTreeClassifier(max_depth=maxDepth, criterion=criterion)
            with mlflow.start_run() as run:
                model = model.fit(features_train, labels_train)
                mlflow.log_metric("val_accuracy", model.score(features_validation, labels_validation))

    # Training and testing of ML-Model with optimal hyperparameters (maxDepth:12, criterion:entropy)
    final_model = train_model(current_best_depth, current_best_criterion, features_train, labels_train)
    testing_score = final_model.score(features_test, labels_test)
    print(testing_score)



def preprocessing1_selection(file_list:list) -> pd.core.frame.DataFrame:
    """
    Function to perform the first step of preprocessing. (Concatenate all files, select relevant rows and columns).

    Parameters
    ----------
    file_list : list
        List containing all files as relative path in directory.

    Returns
    -------
    df : pd.core.frame.DataFrame
        Preprocessed pd.core.frame.DataFrame.dataframe with every row reflecting one player within one game.
    """
    df_result = []

    # Concatenate all csv-files
    for filename in file_list:
        tmpDf = pd.read_csv(filename, index_col=None, header=0)

        # Select necessary columns
        tmpDf = tmpDf[['dateid', 'platform', 'gamemode', 'mapname', 'matchid', 'roundnumber',
                       'objectivelocation', 'winrole', 'skillrank', 'role', 'team', 'haswon', 'operator', 'primaryweapon']]

        # Select only games played on pc with "bomb" gamemode
        tmpDf = tmpDf[(tmpDf['platform'] == 'PC') & (tmpDf['gamemode'] == 'BOMB')]

        '''To minimize the randomizing effect of individual skill only games of the highest ranking players will be 
        selected The assumption is that since everyone knows the game and can play equally well, the operator selection 
        for specific maps and sites become more relevant Select all games where there is at least 1 rank "Diamond" 
        player. In R6 Season 5 you can only queue with players within 1000 skillpoint difference The highest rank in a 
        team is more relevant since one very good player can easily win against 5 bad players in R6 '''
        tmpDf = tmpDf[tmpDf['matchid'].isin(tmpDf.loc[(tmpDf['skillrank'] == 'Diamond'), 'matchid'])]

        # Select necessary columns to further reduce amount of data to be processed
        tmpDf = tmpDf[['dateid', 'mapname', 'matchid', 'roundnumber',
                       'objectivelocation', 'winrole', 'role', 'team', 'haswon', 'operator', 'primaryweapon']]

        # Append selected rows to list
        df_result.append(tmpDf)

    # Concatenate all selected rows into one single dataframe
    df = pd.concat(df_result, axis='rows', ignore_index=True)
    return df

def preprocessing2_logical_df(df:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Function to perform the second step of preprocessing. (Convert result from preprocessing1_selection() into format ready for encoding).

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Return value from preprocessing1_selection().

    Returns
    -------
    df : pd.core.frame.DataFrame
        Preprocessed pd.core.frame.DataFrame.dataframe with every row reflecting one datapoint / one match with all relevant features and label.
        Ready to be encoded.
    """
    # Create a matchID for every single round in a game
    df['realMatchID'] = df['matchid'].astype(str) + "-" + df['roundnumber'].astype(str)

    # Select relevant columns and replace "Attacker" and "Defender" with 0 and 1 for better sorting
    df2 = df[['realMatchID', 'operator', 'role']]
    df2 = df2.replace({'Attacker': 0, 'Defender': 1})

    # Sort the rows by "realMatchID" first and "role" second
    df2 = df2.sort_values(['realMatchID', 'role'], ascending=[True, True])

    # Remove all games that were not played with 5 players on each side (players left during the match)
    df2 = df2.groupby('realMatchID').filter(lambda x: len(x) == 10)

    # Save to csv for testing
    #df2.to_csv("PreprocessingTest.csv", encoding='utf-8', index=False)

    # Create new dataframe with column-structure needed for training
    new_columns = ['realMatchID', 'aop1', 'aop2', 'aop3', 'aop4', 'aop5', 'dop1', 'dop2', 'dop3', 'dop4', 'dop5']
    df_new_columns = pd.DataFrame(columns=new_columns)

    # Set the "realMatchID" column to all unique values
    df_new_columns['realMatchID'] = df2['realMatchID'].unique()
    df_new_columns = df_new_columns.sort_values('realMatchID', ascending=True)

    # Create separate dataframe containing only the maps and sites on which every game took place and the role that won
    df_map_site_win = df[['realMatchID', 'mapname', 'objectivelocation', 'winrole']]

    # Manually put operator values from original dataframe into dataframe with new column structure created in line 25
    for n in df_new_columns['realMatchID']:
        print(n)
        df_original = df2[df2['realMatchID'] == n]

        # aop1
        var = df_original[df_original['role'] == 0].iloc[0]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'aop1'] = var

        # aop2
        var = df_original[df_original['role'] == 0].iloc[1]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'aop2'] = var

        # aop3
        var = df_original[df_original['role'] == 0].iloc[2]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'aop3'] = var

        # aop4
        var = df_original[df_original['role'] == 0].iloc[3]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'aop4'] = var

        # aop5
        var = df_original[df_original['role'] == 0].iloc[4]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'aop5'] = var

        # dop1
        var = df_original[df_original['role'] == 1].iloc[0]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'dop1'] = var

        # dop2
        var = df_original[df_original['role'] == 1].iloc[1]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'dop2'] = var

        # dop3
        var = df_original[df_original['role'] == 1].iloc[2]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'dop3'] = var

        # dop4
        var = df_original[df_original['role'] == 1].iloc[3]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'dop4'] = var

        # dop5
        var = df_original[df_original['role'] == 1].iloc[4]['operator']
        df_new_columns.loc[(df_new_columns['realMatchID'] == n), 'dop5'] = var


    # Merge dataframe containing map, site and winrole with dataframe containing operators
    dfLogicalData = pd.merge(df_new_columns, df_map_site_win, how="left", on="realMatchID")
    dfLogicalData = dfLogicalData.drop_duplicates()

    return dfLogicalData


def one_hot_encode(df:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Function to perform one-hot-encoding of completely preprocessed dataframe.

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Return value from preprocessing2_logical_df().

    Returns
    -------
    df_encoded : pd.core.frame.DataFrame
        Preprocessed and encoded pd.core.frame.DataFrame.dataframe with every row reflecting one datapoint / one match with all relevant features and label.
        Ready to be split.
    """
    df_to_encode = df.replace({'Attacker':0, 'Defender':1}) #Did the Defender win?
    df_to_encode = df_to_encode.drop('realMatchID', axis=1)

    # Use one-hot-encoding on all operators, maps and objective sites
    df_encoded = pd.get_dummies(data=df_to_encode, columns=['aop1', 'aop2', 'aop3', 'aop4', 'aop5', 'dop1', 'dop2', 'dop3', 'dop4', 'dop5', 'mapname', 'objectivelocation'])
    return df_encoded


def split_train_val_test(df_encoded:pd.core.frame.DataFrame) -> tuple[pandas.core.frame.DataFrame, pandas.core.series.Series, pandas.core.frame.DataFrame, pandas.core.series.Series, pandas.core.frame.DataFrame, pandas.core.series.Series]:
    """
    Function to split data into 70% training, 20% validation and 10% testing.

    Parameters
    ----------
    df_encoded : pd.core.frame.DataFrame
        Return value from one_hot_encode().

    Returns
    -------
    features_train : pd.core.frame.DataFrame
        Dataframe containing all features for training.
    labels_train : pandas.core.series.Series
        Series containing all labels for training.
    features_validation : pd.core.frame.DataFrame
        Dataframe containing all features for validation.
    labels_validation : pandas.core.series.Series
        Series containing all labels for validation.
    features_test : pd.core.frame.DataFrame
        Dataframe containing all features for testing.
    labels_test : pandas.core.series.Series
        Series containing all labels for testing.
    """
    # Set labels and features for current dataset
    labels = df_encoded['winrole']
    features = df_encoded.drop('winrole', axis=1)

    # Split data into: training 70%, validation 20%, testing 10%
    features_train, features_validation_test, labels_train, labels_validation_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    features_validation,features_test, labels_validation, labels_test = train_test_split(features_validation_test, labels_validation_test, test_size=0.33, random_state=42)

    return features_train, labels_train, features_validation, labels_validation, features_test, labels_test


def train_model(maxDepth:int, criterion:str, features_train:pandas.core.frame.DataFrame, labels_train:pandas.core.series.Series) -> sklearn.tree.DecisionTreeClassifier:
    """
    Function to train a DecisionTreeClassifier from Scikitlearn with data and 2 parameters.

    Parameters
    ----------
    maxDepth : int
        Maximum depth of DecisionTree
    criterion : str
        Criterion for DecisionTree to decide how to split the data.
    features_train : pd.core.frame.DataFrame
        Dataframe containing all features for training.
    labels_train : pandas.core.series.Series
        Series containing all labels for training.

    Returns
    -------
    model : sklearn.tree.DecisionTreeClassifier
        Trained model ready to perform predictions and to be scored.
    """
    model = tree.DecisionTreeClassifier(max_depth=int(maxDepth), criterion=criterion)
    model = model.fit(features_train, labels_train)
    return model


def hyperparam_tuning(features_train:pandas.core.frame.DataFrame, labels_train:pandas.core.series.Series, features_validation:pandas.core.frame.DataFrame, labels_validation:pandas.core.series.Series) -> tuple[int, str]:
    """
    Function to find the optimal maximum depth and criterion for a sklearn.tree.DecisionTreeClassifier.

    Parameters
    ----------
    features_train : pd.core.frame.DataFrame
        Dataframe containing all features for training.
    labels_train : pandas.core.series.Series
        Series containing all labels for training.
    features_validation : pd.core.frame.DataFrame
        Dataframe containing all features for validation.
    labels_validation : pandas.core.series.Series
        Series containing all labels for validation.

    Returns
    -------
    current_best_depth : int
        Optimal maximum depth of DecisionTree
    current_best_criterion : str
        Optimal criterion for DecisionTree
    """
    # Set default values
    current_best_score = 0
    current_best_depth = 0
    current_best_criterion = 'default'
    criterion_list = ['gini', 'entropy', 'log_loss']

    '''
    Perform gridsearch on all possible parameters maxDepth < 20000 since there are around 20000 points of data and
    setting maxDepth = number of datapoints is overfitting.
    '''
    for criterion in criterion_list:
        for n in range(1, 20001):
            varScore = train_model(n, criterion, features_train, labels_train).score(features_validation, labels_validation)
            if(varScore > current_best_score):
                current_best_score = varScore
                current_best_depth = n
                current_best_criterion = criterion

    return current_best_depth, current_best_criterion


if __name__ == '__main__':
    main()
