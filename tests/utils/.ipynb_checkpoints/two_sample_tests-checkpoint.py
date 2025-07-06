import scipy
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore", message="p-value capped")
warnings.filterwarnings("ignore", message="p-value floored")


""" ________ Settings ________ """

tf.random.set_seed(0)
np.random.seed(0)


""" ________ C2ST ________"""

def C2ST(
        df1 : pd.DataFrame, 
        df2 : pd.DataFrame, 
        mode : str='nn', 
        split : float=0.7, 
        verbose : int=0
) -> tuple[float, float]:

    
    """ 
    Classifier Two-Sample Test (C2ST)


    Trains a binary classifier to distinguish between two datasets (df1 and df2). 
    If the classifier achieves high accuracy, it indicates that the datasets likely come 
    from different distributions. More details: https://arxiv.org/pdf/1610.06545

 
    Args:
    - df1 (pandas dataFrame): The first dataset.
    - df2 (pandas dataFrame): The second dataset.
    - mode (string, optional): The classification model to use.
         Options:
          - 'lr'  : Logistic Regression
          - 'svc' : Support Vector Classifier
          - 'nn'  : Neural Network (default)
    - split (float, optional): The proportion of the dataset used for training (range: (0,1), default is 0.7).
    - verbose (int, optional): Verbosity level for model training (only applies to 'nn' mode, default is 0).


    Return:
        tuple[float, float]:

        - statistic (float): The classification accuracy on the test set, indicating how well 
            the model distinguishes between the two datasets. A  higher value suggests stronger 
          evidence against the null hypothesis. Typically ranges from 0.5 (random guessing) to 1 (perfect classification).
        - null_p_value (float): The probability of obtaining an accuracy at least as high as 
          statistic under the assumption that df1 and df2 come from the same distribution. 
          Lower values indicate stronger evidence that the distributions are different.
    """

    x = df1.copy()
    y = df2.copy()
    if 'id' not in x.columns:
        x['id'] = np.zeros(shape=(len(x)), dtype='int')
    if 'id' not in y.columns:
        y['id'] = np.ones(shape=(len(y)), dtype='int')

    # 1. concatenate samples
    xy = pd.concat([x, y]).sample(frac=1, ignore_index=True)

    # 2. split dataset
    xy_train = xy.loc[:int(split*len(xy)), :]
    xy_test = xy.loc[int(split*len(xy)):, :]

    if mode=='lr':

        # 3. train classifier
        classifier = LogisticRegression(random_state=0).\
            fit(xy_train.drop('id', axis=1).values, xy_train['id'].values)

        # 4. test classifier
        statistic = classifier.score(xy_test.drop('id', axis=1).values, xy_test['id'].values)

    elif mode=='svc':

        # 3. train classifier
        classifier = SVC(random_state=0).fit(xy_train.drop('id', axis=1).values, xy_train['id'].values)

        # 4. test classifier
        statistic = classifier.score(xy_test.drop('id', axis=1).values, xy_test['id'].values)
    
    elif mode=='nn':

        # 3. train classifier
        epochs = 50            # consider parameterizing these
        h = 20
        
        x_tr = tf.convert_to_tensor(xy_train.drop('id', axis=1).values)
        y_tr = tf.convert_to_tensor(xy_train['id'].values)
        x_te = tf.convert_to_tensor(xy_test.drop('id', axis=1).values)
        y_te = tf.convert_to_tensor(xy_test['id'].values)

        classifier = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(x_tr.shape[1],)),
                tf.keras.layers.Dense(units=h, activation="relu", name="layer1"),
                tf.keras.layers.Dense(units=1, activation='sigmoid', name="layer2"),
            ]
        )
        classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
        classifier.fit(x_tr, y_tr, epochs=epochs, verbose=verbose)

        # 4. test classifier
        _, statistic = classifier.evaluate(x_te, y_te, verbose=verbose)

    # 5. p-value
    null_p_value = 1 - scipy.stats.norm.cdf(statistic, loc=1/2, scale=np.sqrt(0.25/len(xy_test)))
    # scipy.stats.norm.rvs(size=(1), loc=1/2, scale=1/(4*len(xy_test)))

    return statistic, null_p_value
