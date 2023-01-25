# coding=utf-8
from joblib import load

from flask import Flask, jsonify, request, jsonify, render_template
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

#!/usr/bin/env python
# -*- coding: utf-8 -*-

app = Flask(__name__)


# Chargement les données
data_train = pd.read_csv("application_train_red.csv")
data_test = pd.read_csv("application_test.csv")

train = None
test = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# Apprentissage du modèle
@app.route("/init_model", methods=["GET"])
def init_model():

    # préparation des données
    df_train, df_test = features_engineering(data_train, data_test)

    print("Features engineering done")
    # préprocessing des données
    df_train, df_test = preprocesseur(df_train, df_test)

    # transformation du dataset de test préparé en variabe globale, car il est utilisé dans la fonction predict
    global train
    train = df_train.copy()

    global test
    test = df_test.copy()

    print("Preprocessing done")
    # resampling des données d'entraînement
    X, y = data_resampler(df_train, data_train)
    print("Resampling done")

    # entraînement du modèle et on le transforme en variable globale pour la fonction predict
    global clf_rfc
    clf_rfc = entrainement_randomforest(X, y)
    print("Training RandomForest done")

    global knn
    knn = entrainement_knn(df_train)
    print("Training knn done")

    return jsonify([" "])


# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():

    return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target():

    df_target = data_train["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]

    print(data_client)
    dict_infos = {
       "status_famille" : data_client["NAME_FAMILY_STATUS"].item(),
       "nb_enfant" : data_client["CNT_CHILDREN"].item(),
       "age" : int(data_client["DAYS_BIRTH"].values / -365),
       "revenus" : data_client["AMT_INCOME_TOTAL"].item(),
       "montant_credit" : data_client["AMT_CREDIT"].item(),
       "annuites" : data_client["AMT_ANNUITY"].item(),
       "montant_bien" : data_client["AMT_GOODS_PRICE"].item()
       }

    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():

    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():

    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]

    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)

    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

@app.route("/predict", methods=["GET"])
def predict():

    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])

    data_client = test[index]

    print(data_client)

    prediction = clf_rfc.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)

@app.route("/load_neighbors", methods=["GET"])
def load_neighbors():

    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test[index]

    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_neighbors = data_train.iloc[indices[0], :]

    response = json.loads(df_neighbors.to_json(orient='index'))

    return response


def features_engineering(data_train, data_test):

    # LABEL ENCODING : Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Itération à chaque colonne
    for col in data_train:
        if data_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(data_train[col])
                # Transform both training and testing data
                data_train[col] = le.transform(data_train[col])
                data_test[col] = le.transform(data_test[col])

                # Keep track of how many columns were label encoded
                le_count += 1


    # ONE HOT ENCODING :

    #one-hot encoding sur les variabes catégorielles
    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)

    train_labels = data_train['TARGET']
    # Alignement des datas "train" et "test", On garde les colonnes présentes dans les deux dataframes.
    data_train, data_test = data_train.align(data_test, join = 'inner', axis = 1)
    # On rajoute la cible
    data_train['TARGET'] = train_labels


    # VALEURS ABERRANTES

    # On créé une colonne d'indicateur d'anomalie (flag)
    data_train['DAYS_EMPLOYED_ANOM'] = data_train["DAYS_EMPLOYED"] == 365243
    # On remplace les valeurs anormales avec des "Nan"
    data_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    data_test['DAYS_EMPLOYED_ANOM'] = data_test["DAYS_EMPLOYED"] == 365243
    data_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

    # Traitement des valeurs négatives
    data_train['DAYS_BIRTH'] = abs(data_train['DAYS_BIRTH'])


    # CREATION DE VARIABLES

    # Ces fonctionnalités sont repris du kaggle d'Aguiar :

    #CREDIT_INCOME_PERCENT : le pourcentage du montant du crédit par rapport au revenu d'un client
    # ANNUITY_INCOME_PERCENT : le pourcentage de l'annuité du prêt par rapport au revenu d'un client
    # CREDIT_TERM : la durée du paiement en mois (puisque l'annuité est le montant mensuel dû
    # DAYS_EMPLOYED_PERCENT : le pourcentage de jours employés par rapport à l'âge du client

    data_train_domain = data_train.copy()
    data_test_domain = data_test.copy()

    data_train_domain['CREDIT_INCOME_PERCENT'] = data_train_domain['AMT_CREDIT'] / data_train_domain['AMT_INCOME_TOTAL']
    data_train_domain['ANNUITY_INCOME_PERCENT'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_INCOME_TOTAL']
    data_train_domain['CREDIT_TERM'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_CREDIT']
    data_train_domain['DAYS_EMPLOYED_PERCENT'] = data_train_domain['DAYS_EMPLOYED'] / data_train_domain['DAYS_BIRTH']

    data_test_domain['CREDIT_INCOME_PERCENT'] = data_test_domain['AMT_CREDIT'] / data_test_domain['AMT_INCOME_TOTAL']
    data_test_domain['ANNUITY_INCOME_PERCENT'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_INCOME_TOTAL']
    data_test_domain['CREDIT_TERM'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_CREDIT']
    data_test_domain['DAYS_EMPLOYED_PERCENT'] = data_test_domain['DAYS_EMPLOYED'] / data_test_domain['DAYS_BIRTH']

    return data_train_domain, data_test_domain

def preprocesseur(df_train, df_test):

    # Cette fonction permet d'imputer les valeurs manquantes dans chaque dataset et aussi d'appliquer un MinMaxScaler

    # On supprime la "TRAGET" des données "Train"
    if "TARGET" in df_train:
        train = df_train.drop(columns = ["TARGET"])
    else:
        train = df_train.copy()

    # Feature names
    features = list(train.columns)


    # Imputation médiane des valeurs manquantes
    imputer = SimpleImputer(strategy = 'median')

    # Mettre à l'échelle chaque feature à 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # On remplace la colonne booléenne par des valeurs numériques
    train["DAYS_EMPLOYED_ANOM"] = train["DAYS_EMPLOYED_ANOM"].astype("int")

    # Fit sur les données "Train"
    imputer.fit(train)

    # Transformez les données de "Train""et "test"
    train = imputer.transform(train)
    test = imputer.transform(df_test)

    # On répète avec le scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test

def data_resampler(df_train, target):

    rsp = RandomUnderSampler()
    X_rsp, y_rsp = rsp.fit_resample(df_train, target["TARGET"])

    return X_rsp, y_rsp

def entrainement_randomforest(X, y):

    # Configuration optimum trouvée par le GridSearchCV
    clf_rfc = RandomForestClassifier(
                                      n_estimators= 10,
                                      max_depth = 2,
                                      random_state = 0,
                                      max_samples = .15
                                    )



    clf_rfc.fit(X, y)

    return clf_rfc

def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    #app.run(host='172.31.45.211', port=8080)
