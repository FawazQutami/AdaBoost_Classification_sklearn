# File name: AdaBoost_Classification_sklearn.py

import pandas as pd
import numpy as np
from time import perf_counter

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore")
marks = '-' * 100


def confusion_matrix_accuracy(cm):
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    return diagonal_sum / sum_of_all_elements


def validation(model, x_trn, y_trn, random_state):
    # Grid parameters
    grid_params = dict(n_estimators=[10, 50, 100, 500]
                       , learning_rate=[0.0001, 0.001, 0.01, 0.1])

    # Evaluation procedure
    c_v = RepeatedKFold(n_splits=10
                        , n_repeats=3
                        , random_state=random_state)

    # Grid search procedure
    grid_search = GridSearchCV(estimator=model
                               , param_grid=grid_params
                               , n_jobs=-1
                               # -1 means using all processors
                               , cv=c_v
                               , scoring='accuracy'
                               , error_score='raise'
                               )
    # Fit the grid search
    results = grid_search.fit(x_trn, y_trn)
    # Best score - Mean cross-validated score of the best_estimator
    print("-- GridSearchCV best score result:\n\t %.3f%%" % (results.best_score_ * 100))
    # Best estimator - Estimator that was chosen by the search
    print("-- GridSearchCV best estimator result:\n\t %s" % results.best_estimator_)

    return results.best_params_['learning_rate'], results.best_params_['n_estimators']


def ada_boost_classification(x_cls, y_cls, random_state, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(x_cls, y_cls
                                                        , test_size=0.2
                                                        , random_state=random_state)
    # Scale the features
    X_train, X_test = scaling_data(X_train, X_test)
    """
        When using machine learning algorithms that have a stochastic learning algorithm, it is good practice 
        to evaluate them by averaging their performance across multiple runs or repeats of cross-validation. 
        When fitting a final model it may be desirable to either increase the number of trees until the variance 
        of the model is reduced across repeated evaluations, or to fit multiple final models and average 
        their predictions.
        """
    # Evaluate the model using Grid Search
    learning_rate, n_estimators = validation(AdaBoostClassifier()
                                             , X_train
                                             , y_train
                                             , random_state=random_state)

    # Build the AdaBoostClassifier using the best parameters from the grid search
    base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    ab = AdaBoostClassifier(base_estimator=base_estimator
                            , n_estimators=n_estimators
                            , learning_rate=learning_rate
                            , random_state=random_state)
    # Fit
    ab.fit(X_train, y_train)
    print('-- Model parameters/configuration:\n', ab.get_params())
    # Predict
    y_predicted = ab.predict(X_test)

    # Report the model performance
    # Performance report for testing data ---------------------
    test_data_list = performance_report(y_test, y_predicted)
    # Performance report for training data ---------------------
    train_data_list = performance_report(y_train, ab.predict(X_train))

    metrics = ['Accuracy Score', 'Confusion Matrix Accuracy', 'F1 SCORE for test set']
    performance_df = pd.DataFrame({'Training performance': train_data_list
                                      , 'Testing performance': test_data_list}
                                  , index=metrics)
    print('\n-- Performance report:\n', performance_df)

    # Get the feature importance ------------
    feature_imp = pd.DataFrame({'importance': ab.feature_importances_}
                               , index=list(x_cls))
    print('\n-- Feature Importance:\n', feature_imp.sort_values(by=['importance']
                                                                , ascending=False))

    return feature_imp.nlargest(10, 'importance').sort_values(by=['importance']
                                                              , ascending=False)


def performance_report(set1, set2):
    acc = '{:.2f}'.format(accuracy_score(set1, set2) * 100.0)
    cm = '{:.2f}'.format(confusion_matrix_accuracy(confusion_matrix(set1, set2)) * 100.0)
    f1 = '{:.2f}'.format(f1_score(set1, set2, average='weighted') * 100.0)

    performance_list = [acc, cm, f1]

    return performance_list


def label_encoders(labels, flag):
    """ Encode target labels with value between 0 and n_classes-1. """
    label_encoder = LabelEncoder()
    # print(label_encoder.classes_) # to get the encoded labels
    if flag == 'transform':
        # Fit label encoder and return encoded labels.
        encoded_labels = label_encoder.fit_transform(labels)
    else:
        # Transform labels back to original encoding.
        encoded_labels = label_encoder.inverse_transform(labels)

    return encoded_labels


def scaling_data(x_train, x_test):
    """ Scaling or standardizing our training and test data """
    """
        -- Data standardization is the process of rescaling the attributes so that they have 
            mean as 0 and variance as 1.
        -- The ultimate goal to perform standardization is to bring down all the features to 
            a common scale without distorting the differences in the range of the values.
        -- In sklearn.preprocessing.StandardScaler(), centering and scaling happens independently 
            on each feature.
    """
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance
    """
        The "fit method" is calculating the mean and variance of each of the features present in our data. 
        The "transform" method is transforming all the features using the respective mean and variance.
    """
    scaled_x_train = scaler.fit_transform(x_train)
    """
        Using the "transform" method we can use the same mean and variance as it is calculated from our 
        training data to transform our test data. Thus, the parameters learned by our model using the 
        training data will help us to transform our test data.
    """
    scaled_x_test = scaler.transform(x_test)
    return scaled_x_train, scaled_x_test


def get_data():
    features, target, data_name = None, None, None

    flag = int(input("\nChoose a Dataset: *****************************"
                     "\n(1) Iris Dataset "
                     "\n(2) Brest cancer Dataset "
                     #"\n(3) Boston house-prices Dataset "
                     "\nExit: press any other key!\n"))
    if flag == 1:
        # Load Iris dataset
        iris = datasets.load_iris()
        data = pd.DataFrame(data=np.c_[iris['data'], iris['target']]
                            , columns=np.append(iris['feature_names'], ['target']))
        features, target = data.drop(columns=['target'], axis=1), data['target']
        # Desired outputs y1, ..., yn ∈{-1, 1}
        # target[target == 0] = -1
        data_name = 'Iris dataset'

    elif flag == 2:
        # Load Brest Cancer dataset
        bc = datasets.load_breast_cancer()
        data = pd.DataFrame(data=np.c_[bc['data'], bc['target']]
                            , columns=np.append(bc['feature_names'], ['target_names']))
        features, target = data.drop(columns=['target_names'], axis=1), data['target_names']
        data_name = 'Brest Cancer dataset'
        """
    elif flag == 3:
        # Load Boston house-prices dataset
        bh = datasets.load_boston()
        # features, target =bh.data, bh.target
        data = pd.DataFrame(data=np.c_[bh['data'], bh['target']]
                            , columns=np.append(bh['feature_names'], ['MEDV']))
        features, target = data.drop(columns=['MEDV'], axis=1), data['MEDV']
        # Convert continuous target
        target = label_encoders(target, 'transform')
        data_name = 'Boston house-prices dataset'"""

    else:
        print(marks)
        print("Exit")
        print(marks)
        exit()

    return features, target, data_name


def plot_feature_importance(fi, nam):
    x_labels = list(fi.index)
    y_labels = list(fi['importance'])
    x = np.arange(1, len(x_labels) + 1, 1)

    plt.style.use('ggplot')
    # Format the plot
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.5

    ax.bar(x, fi['importance'], width)
    plt.axis([0, max(x) + width, 0, max(y_labels) + 0.1])
    ax.axhline(y=0, color='g')
    ax.axvline(x=0, color='g')
    ax.set_xlabel('Features', fontdict=font)
    ax.set_ylabel('Importance', fontdict=font)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    color = ['c', 'g', 'b', 'm']
    for i in x:
        ax.hlines(y=y_labels[i - 1]
                  , xmin=min(x) - 1
                  , xmax=i
                  , colors='b'
                  , linestyles='dashed')
        ax.text(x=i
                , y=y_labels[i - 1]
                , s='{:.4f}'.format(y_labels[i - 1])
                , ha='center'
                , va='bottom'
                , color='darkred'
                , fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Feature importance for ' + nam, fontdict=font)
    plt.tight_layout(True)
    plt.show()


if __name__ == '__main__':
    while True:
        X, y, name = get_data()

        # AdaBoost Classification
        print(marks)
        print('AdaBoost Classification for ' + name + f'\n\tX.shape: {X.shape}, y.shape: {y.shape}')
        print(marks)

        start = perf_counter()
        feature_importance = ada_boost_classification(X, y, random_state=5, max_depth=10)
        end = perf_counter()

        execution_time = end - start
        print(f"\n-- Execution time:\n\t {execution_time: .1f} seconds --> {execution_time / 60: .1f} minutes")

        print(marks)
        plot_feature_importance(feature_importance, name)
