from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, classification_report
from dython import nominal
import sweetviz
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from numpy import mean
from numpy import std
import numpy as np
from sklearn.model_selection import cross_val_score
import seaborn as sn
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv(
    "D:/CME4403/term_project/dataset/train.csv", na_values=[np.nan])
# Delete 'Track Name' feature, redundant
dataset = dataset.drop(['Track Name', 'Artist Name'], axis=1)

# rename feature names
dataset = dataset.rename(columns={'Popularity': 'popularity',
                                  'duration_in min/ms': 'duration', 'Class': 'music genre'})
# relationship between both categorical and numerical features
nominal.associations(dataset, figsize=(30, 15), mark_columns=True)

class_map = {0: 'Acoustic/Folk', 1: 'Alt_Music', 2: 'Blues', 3: 'Bollywood', 4: 'Country', 5: 'HipHop',
             6: 'IndieAlt', 7: 'Instrumental', 8: 'Metal', 9: 'Pop', 10: 'Rock'}

dataset['music genre'] = dataset['music genre'].map(class_map)
"""
# get dataset analyze report
report = sweetviz.analyze(dataset)
report.show_html()
"""
sn.factorplot(data=dataset, kind='count', aspect=3, size=5, x='music genre')
###############################################################################

def createNewFeatureFromTarget(dataset):
        """
        #for the new created feature : 'energic_music'
        dataset_cat = pd.get_dummies(dataset['music genre'])
        dataset_final = pd.concat([dataset, dataset_cat], axis=1)
        dataset_final = dataset_final.drop(['music genre'], axis=1)
        """
        energic_music = []
        for genre in dataset['music genre']:
            if (genre == 'Acoustic/Folk' ):
                energic_music.append('1')
            elif (genre == 'Alt_Music'):
                energic_music.append('0')
            elif (genre == 'Blues' ):
                energic_music.append('1')
            elif (genre == 'Bollywood' ):
                energic_music.append('1')
            elif (genre == 'Country' ):
                energic_music.append('0')
            elif (genre == 'HipHop' ):
                energic_music.append('0')
            elif (genre == 'IndieAlt' ):
                energic_music.append('0')
            elif (genre == 'Instrumental'):
                energic_music.append('1')
            elif (genre == 'Metal' ):
                energic_music.append('1')
            elif (genre == 'Pop' ):
                energic_music.append('0')
            else:
                energic_music.append('1')

        dataset['energic_music'] = energic_music
        dataset = dataset.drop(['music genre'], axis=1)
        sn.factorplot(data=dataset, kind='count', aspect=3, size=5, x='energic_music')
        return dataset

###############################################################################
"""
# check for missing values and duplicates
print(dataset.isnull().sum()/len(dataset))
sn.heatmap(pd.DataFrame(dataset.isnull().sum()/len(dataset), columns=['data']))
plt.show()
"""
# fill nan values with median
df = dataset[dataset.columns[dataset.isna().any()]]
for row in list(df.columns.values):
    dataset[row].fillna((dataset[row].median()), inplace=True)
"""
# check nan value changes
sn.heatmap(pd.DataFrame(dataset.isnull().sum()/len(dataset), columns=['data']))
plt.show()
"""
duplicate = dataset[dataset.duplicated()]
dataset.drop_duplicates(inplace=True)

###############################################################################
#normalization - loudness-tempo-popularity-duration
dataset.loudness = (dataset.loudness - dataset.loudness.min()) / \
    (dataset.loudness.max() - dataset.loudness.min())
dataset.tempo = (dataset.tempo - dataset.tempo.min()) / \
    (dataset.tempo.max() - dataset.tempo.min())
dataset.popularity = (dataset.popularity - dataset.popularity.min()) / \
    (dataset.popularity.max() - dataset.popularity.min())
dataset.duration = (dataset.duration - dataset.duration.min()) / \
    (dataset.duration.max() - dataset.duration.min())
###############################################################################
#get discrete and continuous features dataset
numerical = dataset.select_dtypes(include=['float64', 'int64'])
discrete = dataset.select_dtypes(include=['float64', 'int64'])
continuous = dataset.select_dtypes(include=['float64', 'int64'])

numeric_discrete = ['key', 'mode', 'time_signature']

for ftr in list(numeric_discrete):
    continuous = continuous.drop([ftr], axis=1)

numeric_continuous = ['popularity', 'danceability', 'energy', 'loudness',
                      'speechiness', 'acousticness','instrumentalness',
                      'liveness', 'valence', 'tempo','duration']

for ftr in list(numeric_continuous):
    discrete = discrete.drop([ftr], axis=1)
###############################################################################
#find and handle with outliers for continuous features
def find_outliers(feature):
    q1, q3 = np.percentile(feature, [25, 75])
    iqr = q3 - q1
    min_thresold = q1 - (iqr * 1.5)
    max_thresold = q3 + (iqr * 1.5)
    feature.loc[feature > max_thresold] = max_thresold
    feature.loc[feature < min_thresold] = min_thresold

for ftr in list(continuous.columns.values):
    find_outliers(dataset[ftr])
    
# relationship between both categorical and numerical features
nominal.associations(dataset, figsize=(30, 15), mark_columns=True)

###############################################################################•

def evaluation_result(model_predictions,y_test):
    print()
    print("Accuracy: ", accuracy_score(y_test, model_predictions))
    print(classification_report(y_test, model_predictions))
    mae = metrics.mean_absolute_error(y_test, model_predictions)
    mse = metrics.mean_squared_error(y_test, model_predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, model_predictions)
    print("Results of sklearn.metrics:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R-Squared:", r2)

def model_implement(model,x_train, y_train,x_test,y_test):
    model.fit(x_train, y_train)
    model_predictions = model.predict(x_test)
    evaluation_result(model_predictions,y_test)
    
###############################################################################

def firstPart(dataset):
    # encoding categorical features
    mapping = {"music genre": {"Acoustic/Folk": 0,
                                  "Alt_Music": 1, 
                                  "Blues": 2,
                                  "Bollywood": 3,
                                  "Country": 4,
                                  "HipHop": 5,
                                  "IndieAlt": 6,
                                  "Instrumental": 7,
                                  "Metal": 8,
                                  "Pop": 9,
                                  "Rock": 10}}
    dataset = dataset.replace(mapping)
    
    #define target and dataset
    target = dataset['music genre']
    dataset = dataset.drop(['music genre'], axis=1)
    random_state = 18

    # Test technic --> %70 train, %30 test
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, target, test_size=0.3, random_state=random_state)
    print("Part 1 - Random Forest")
    # Random Forest
    model = RandomForestClassifier(n_estimators=1000,criterion ="entropy", n_jobs =-1)
    model_implement(model,x_train, y_train,x_test,y_test)
    print("Part 1 - Neural Network")
    #neural network
    snn_classifier = MLPClassifier(hidden_layer_sizes = 100,activation ="relu",
                                   max_iter = 500, solver="adam",shuffle = True)
    model_implement(snn_classifier,x_train, y_train,x_test,y_test)
    print("Part 1 - Logistic Regression")
    # define the multinomial logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='sag',max_iter=1500,C=100)
    model_implement(model,x_train, y_train,x_test,y_test)
    print("Part 1 - Support Vector Machine")
    #Support Vector Machine
    model = SVC(C=10, kernel = "linear")
    ovo = OneVsOneClassifier(model)
    model_implement(ovo,x_train, y_train,x_test,y_test)
    
###############################################################################
def secondPart(dataset):
    #define target and dataset
    target = dataset['energic_music']
    dataset = dataset.drop(['energic_music'], axis=1)
    random_state = 18

    # Test technic --> %70 train, %30 test
    x_train, x_test, y_train, y_test = train_test_split(
        dataset, target, test_size=0.2, random_state=random_state)
    print("Part 2 - Random Forest")
    # Random Forest
    model = RandomForestClassifier(n_estimators=500, bootstrap=False, criterion="entropy")
    model_implement(model,x_train, y_train,x_test,y_test)
    print("Part 2 - Neural Network")
    #neural network
    snn_classifier = MLPClassifier(max_iter=400)
    model_implement(snn_classifier,x_train, y_train,x_test,y_test)
    print("Part 2 - Logistic Regression")
    #Logistic Regression
    model = LogisticRegression(solver='saga')
    model_implement(model,x_train, y_train,x_test,y_test)
    print("Part 2 - Support Vector Machine")
    #Support Vector Machine
    model = SVC()
    model_implement(model,x_train, y_train,x_test,y_test)
  
firstPart(dataset)
df = createNewFeatureFromTarget(dataset)
secondPart(df)

###############################################################################
"""
# relationship between both categorical and numerical features
nominal.associations(dataset, figsize=(30, 15), mark_columns=True)
report = sweetviz.analyze(dataset)
report.show_html()
"""