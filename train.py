import numpy as np
import pandas as pd
import multiprocessing
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler


SEED = 123
TEST_SIZE = 0.3
CV = 5
SCORING = 'neg_log_loss'
N_JOBS = multiprocessing.cpu_count() - 3
N_ITER = 2500


def to_csv(df, path):
    df.loc[-1] = df.dtypes
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df.to_csv(path, index=False)


def read_csv(path):
    dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    return pd.read_csv(path, dtype=dtypes, skiprows=[1]).set_index("index")


def load_data(type_=None):
    if type_ is None:
        return read_csv(r"data/X.csv"), \
            read_csv(r"data/y.csv")


def train_test_split_data(type_=None):

    X, y = load_data(type_=type_)

    vars = pd.read_excel(r"data\vars.xlsx", engine='openpyxl')
    vars = vars[vars['discard'] == 0]['variable'].values.tolist()
    X = X.loc[:, [var for var in vars if var in X.columns]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y, shuffle=True)
    return X_train, X_test, y_train, y_test


def train_regression(save=True, name=None, type_=None):
    """Hace un grid search de un L1 Logistic Regression Classifier.
    """
    # cargamos data
    X_train, X_test, y_train, y_test = train_test_split_data(type_=type_)
    # hacemos train test split
    log_parameter_grid = {'C': np.logspace(-1.5, 3, 10),
                          'penalty': ['l2', 'l1']}
    estimator = LogisticRegression(random_state=SEED, solver='liblinear', max_iter=20000)
    logmodel = GridSearchCV(estimator=estimator, param_grid=log_parameter_grid,
                            cv=CV, n_jobs=N_JOBS, scoring=SCORING, verbose=0)
    logmodel.fit(X_train, y_train.squeeze())
    best_estimator = logmodel.best_estimator_
    best_estimator.set_params(**logmodel.best_params_)
    best_estimator.set_params(**{'warm_start': True})
    best_estimator.fit(X_train, y_train.squeeze())

    logprobas = best_estimator.predict_proba(X_train)[:, 1]
    metric1 = roc_auc_score(y_train, logprobas)
    metric2 = average_precision_score(y_train, logprobas)
    print("Regresión Logística Train--> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))

    logprobas = best_estimator.predict_proba(X_test)[:, 1]
    metric1 = roc_auc_score(y_test, logprobas)
    metric2 = average_precision_score(y_test, logprobas)
    print("Regresión Logística Test--> AUC: {:.2f}; PR: {:.2f}".format(metric1*100, metric2*100))

    if save:
        tosave = {
            "model": best_estimator,
            "probas": logprobas
        }
        if name is None:
            name = SEED
        with open('data/models/logreg/mod{}.pkl'.format(name), 'wb') as handle:
            pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    else:
        return logmodel, logprobas


def train_random_forest(save=True, name=None, type_=None):
    """Hace un random search de un Random Forest Classifier.
    """
    # cargamos data
    X_train, X_test, y_train, y_test = train_test_split_data(type_=type_)
    rf = RandomForestClassifier(random_state=SEED)
    rf_parameter_grid = {'min_samples_leaf': [2, 4, 8, 16, 32, 64],
                         'n_estimators': [50, 100, 200, 300, 400, 500],
                         'max_features': ['sqrt', 'log2', 'auto'],
                         "max_samples": [0.6, 0.7, 0.8, 0.9, None],
                         "min_samples_split": [2, 4, 8, 16, 32],
                         "max_depth": [2, 4, 8, 16, 32, None]}
    rfmodel = RandomizedSearchCV(
        estimator=rf, param_distributions=rf_parameter_grid, scoring=SCORING,
        cv=CV, n_jobs=N_JOBS, verbose=0, n_iter=N_ITER)

    weights = compute_class_weight('balanced',
                                   np.unique(y_train),
                                   y_train.values.reshape((-1,)))
    weights = y_train.replace({'recontact': dict(zip([0, 1], weights))}).values.ravel()
    rfmodel.fit(X_train, y_train.squeeze())  # , sample_weight=weights
    rfmodel = RandomForestClassifier(**rfmodel.best_params_, random_state=SEED)
    rfmodel.fit(X_train, y_train.squeeze())  # , sample_weight=weights
    rfprobas = rfmodel.predict_proba(X_test)[:, 1]
    metric1 = roc_auc_score(y_test, rfprobas)
    metric2 = average_precision_score(y_test, rfprobas)
    print("Random Forest Test--> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))

    metric1 = roc_auc_score(y_train, rfmodel.predict_proba(X_train)[:, 1])
    metric2 = average_precision_score(y_train, rfmodel.predict_proba(X_train)[:, 1])
    print("Random Forest Train--> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))

    print(rfmodel)
    if save:
        tosave = {
            "model": rfmodel,
            "probas": rfprobas
        }
        if name is None:
            name = SEED
        with open('data/models/randfor/mod{}.pkl'.format(name), 'wb') as handle:
            pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    else:
        return rfmodel, rfprobas


def train_xgboost(save=True, name=None, type_=None):
    """Hace un random search de un XGBoost.
    """
    # cargamos data
    X_train, X_test, y_train, y_test = train_test_split_data(type_=type_)
    xgb = XGBClassifier(random_state=SEED, booster='gbtree',
                        objective='binary:logistic',
                        eval_metric=['logloss'])
    xgb_parameter_grid = {'gamma': np.linspace(0.05, 1.5, 10),
                          'max_depth': [2, 3, 4, 5, 6],
                          'n_estimators': [100, 300, 500],
                          'reg_alpha': np.linspace(1, 11, 20),
                          'reg_lambda': np.linspace(1, 11, 25),
                          "base_score": np.linspace(0.1, 0.6, 10),
                          "subsample": np.arange(0.5, 1, 0.05),
                          "colsample_bytree": np.arange(0.5, 1, 0.05),
                          "learning_rate": [0.1, 0.05]}

    xgbmodel = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_parameter_grid, n_iter=N_ITER,
                                  scoring=SCORING, cv=CV, n_jobs=N_JOBS, verbose=0, random_state=SEED)
    xgbmodel.fit(X_train, y_train.squeeze())
    best_parameters = xgbmodel.best_params_
    xgbmodel = XGBClassifier(random_state=SEED, booster='gbtree',
                             objective='binary:logistic',
                             eval_metric=['logloss'],
                             **best_parameters)
    xgbmodel.fit(X_train, y_train.squeeze())

    xgbprobastr = xgbmodel.predict_proba(X_train)[:, 1]
    metric1 = roc_auc_score(y_train, xgbprobastr)
    metric2 = average_precision_score(y_train, xgbprobastr)
    print("XGBoost Train --> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))
    print(xgbmodel)

    xgbprobas = xgbmodel.predict_proba(X_test)[:, 1]
    metric1 = roc_auc_score(y_test, xgbprobas)
    metric2 = average_precision_score(y_test, xgbprobas)
    print("XGBoost Test --> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))

    print(best_parameters)
    print("\n")
    if save:
        tosave = {
            "model": xgbmodel,
            "probas": xgbprobas
        }
        if name is None:
            name = SEED
        with open('data/models/xgboo/mod{}.pkl'.format(name), 'wb') as handle:
            pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    else:
        return xgbmodel, xgbprobas


def train_svm(save=True, name=None, type_=None):
    """Hace un random search de un XGBoost.
    """
    # cargamos data
    X_train, X_test, y_train, y_test = train_test_split_data(type_=type_)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(probability=True)
    svm_parameter_grid = {
        'C': np.logspace(-1.5, 3, 10),
        'kernel': ['poly', 'rbf']
    }

    svm_model = GridSearchCV(estimator=svm, param_grid=svm_parameter_grid, scoring=SCORING,
                             cv=CV, n_jobs=N_JOBS, verbose=0)
    svm_model.fit(X_train, y_train.squeeze())
    best_parameters = svm_model.best_params_
    svm_model = SVC(probability=True, **best_parameters)
    svm_model.fit(X_train, y_train.squeeze())

    smv_probas_tr = svm_model.predict_proba(X_train)[:, 1]
    metric1 = roc_auc_score(y_train, smv_probas_tr)
    metric2 = average_precision_score(y_train, smv_probas_tr)
    print("SVM Train --> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))
    print(svm_model)

    svm_probas_tt = svm_model.predict_proba(X_test)[:, 1]
    metric1 = roc_auc_score(y_test, svm_probas_tt)
    metric2 = average_precision_score(y_test, svm_probas_tt)
    print("SVM Test --> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))

    print(best_parameters)
    print("\n")
    if save:
        tosave = {
            "model": svm,
            "probas": svm_probas_tt
        }
        if name is None:
            name = SEED
        with open('data/models/svm/mod{}.pkl'.format(name), 'wb') as handle:
            pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    else:
        return xgbmodel, xgbprobas


def train_neighbors(save=True, name=None, type_=None, seed=123):
    """Hace un grid search de un K Neighbors Classifier.
    """
    # cargamos datos
    X_train, X_test, y_train, y_test = train_test_split_data(seed=seed, type_=type_)
    # hacemos train test split
    log_parameter_grid = {'n_neighbors': [3, 5, 7, 10, 15, 20, 30, 50]}
    estimator = KNeighborsClassifier(n_jobs=N_JOBS)
    logmodel = GridSearchCV(estimator=estimator, param_grid=log_parameter_grid,
                            cv=CV, n_jobs=N_JOBS, scoring=SCORING, verbose=0)
    logmodel.fit(X_train, y_train.squeeze())
    best_estimator = logmodel.best_estimator_
    best_estimator.set_params(**logmodel.best_params_)
    best_estimator.fit(X_train, y_train.squeeze())

    logprobas_tr = best_estimator.predict_proba(X_train)[:, 1]
    metric1 = roc_auc_score(y_train, logprobas_tr)
    metric2 = average_precision_score(y_train, logprobas_tr)
    print("K Neighbors Train--> AUC: {:.2f}; PR: {:.2f}".format(metric1 * 100, metric2 * 100))

    logprobas = best_estimator.predict_proba(X_test)[:, 1]
    metric1 = roc_auc_score(y_test, logprobas)
    metric2 = average_precision_score(y_test, logprobas)
    print("K Neighbors Train Test--> AUC: {:.2f}; PR: {:.2f}".format(metric1*100, metric2*100))

    if save:
        tosave = {
            "model": best_estimator,
            "probas": logprobas,
            "probas_tr": logprobas_tr
        }
        if name is None:
            name = seed
        with open('datos/models/kneigh/mod{}.pkl'.format(name), 'wb') as handle:
            pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    else:
        return logmodel, logprobas


if __name__ == '__main__':

    seeds = [0, 8, 10, 23, 37, 42, 69, 99, 123, 747]
    for i, seed in enumerate(seeds):
        SEED = seed
        name = "seed={}".format(seed)
        train_neighbors(save=True, name=name, type_=None)
        train_svm(save=True, name=name, type_=None)
        train_regression(save=True, name=name, type_=None)
        train_random_forest(save=True, name=name, type_=None)
        train_xgboost(save=True, name=name, type_=None)
