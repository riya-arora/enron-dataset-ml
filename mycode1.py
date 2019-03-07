import sys
import cPickle as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data,test_classifier



with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)



df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace('NaN', np.nan, inplace = True)

df.info()

#for key in data_dict.keys():
#   for value1 in data_dict[key]:
#       for value2 in data_dict[key]:
#           if(value1==value2 or value1=='email_address' or value2=='email_address' or data_dict[key][value1]=='None' or data_dict[key][value2]=='None'):
#               continue
#           else:
#               df.plot.scatter(x=value1,y=value2)
#               plt.show()
#   break
df.drop('TOTAL', inplace=True)
#print df['salary'].idxmax()

#df.plot.scatter('salary','shared_receipt_with_poi')
#plt.show()

#print df['restricted_stock'].idxmax()
df['fraction_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['fraction_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']

ax = df[df['poi'] == False].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='blue', label='non-poi')
df[df['poi'] == True].plot.scatter(x='fraction_from_poi', y='fraction_to_poi', color='red', label='poi', ax=ax)
#plt.show()



features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments',
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred',
                 'total_stock_value', 'to_messages', 'from_messages', 'from_this_person_to_poi',
                 'from_poi_to_this_person', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']


filled_df = df.fillna(value='NaN') # featureFormat expects 'NaN' strings
data_dict = filled_df.to_dict(orient='index')
my_dataset = data_dict



data = featureFormat(my_dataset, features_list, sort_keys = True)


y,X=targetFeatureSplit(data)
X=np.array(X)
y=np.array(y)
def evaluate_model(grid, X, y, cv):
    nested_score = cross_val_score(grid, X=X,y=y, cv=cv)
    print "Nested f1 score:" .format(nested_score.mean())

    grid.fit(X,y)
    print "Best parameters:" .format(grid.best_params_)

    cv_accuracy=[]
    cv_precision=[]
    cv_recall=[]
    cv_f1=[]
    for train_index,test_index in cv.split(X,y):
        X_train,X_test=X[train_index],X[test_index]
        Y_train,Y_test=y[train_index],y[test_index]

        grid.best_estimator_.fit(X_train,Y_train)
        pred=grid.best_estimator_.predict(X_test)


        cv_accuracy.append(accuracy_score(Y_test,pred))
        cv_precision.append(precision_score(Y_test,pred))
        cv_recall.append(recall_score(Y_test,pred))
        cv_f1.append(f1_score(Y_test,pred))

    print "Mean Accuracy: {}".format(np.mean(cv_accuracy))
    print "Mean Precision: {}".format(np.mean(cv_precision))
    print "Mean Recall: {}".format(np.mean(cv_recall))
    print "Mean f1: {}".format(np.mean(cv_f1))


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

SCALER = [None, StandardScaler()]
SELECTOR__K = [10, 13, 15, 18, 'all']
REDUCER__N_COMPONENTS = [2, 4, 6, 8, 10]
## pipeline used to create a sequence of events

pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest()),
        ('reducer', PCA(random_state=42)),
        ('classifier', GaussianNB())
    ])

param_grid = {
    'scaler': SCALER,
    'selector__k': SELECTOR__K,
    'reducer__n_components': REDUCER__N_COMPONENTS
}

gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)

evaluate_model(gnb_grid, X, y, sss)

test_classifier(gnb_grid.best_estimator_, my_dataset, features_list)

