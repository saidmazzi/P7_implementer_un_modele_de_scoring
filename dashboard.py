# coding=utf-8
import streamlit as st

import requests
import json

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from flask import Flask
#!/usr/bin/env python
# -*- coding: utf-8 -*-

st.set_option('deprecation.showPyplotGlobalUse', False)

#URL_API= "http://localhost:5000/""
#URL_API = "http://13.36.160.181:80/"
URL_API = "http://ec2-13-36-160-181.eu-west-3.compute.amazonaws.com:8080/"
#URL_API = "http://172.31.45.211:8080/"

def main():

    #init = st.markdown(init_api())
    st.markdown(init_api())
    # Titre et du sous-titre
    st.title("Tableau de Board Client")
    st.subheader("Informations descriptives relatives au client et à un groupe de clients similaires.")

    # Informations dans la sidebar
    st.sidebar.title("OpenClassrooms : Implémenter un modèle de scoring")
    st.sidebar.subheader("Indications générales")
    # Chargement du logo
    logo = load_logo()
    st.sidebar.image(logo,
                     width=200)

    # Chargement de la selectbox
    lst_id = load_selectbox()
    global id_client
    id_client = st.sidebar.selectbox("ID Client", lst_id)

    # Présentation des infos générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen()

    # Présentation des infos dans la sidebar
    # Nombre de crédits existants
    st.sidebar.markdown("<u>Nombre crédits existants dans la base de données :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Graphique "Pie"
    st.sidebar.markdown("<u>aCapacité pour un client à payer ses dettes!</u>", unsafe_allow_html=True)

    plt.pie(targets, explode=[0, 0.1], labels=["Clients fiables", "Clients non fiables"], autopct='%1.1f%%',
            shadow=True, startangle=90)
    st.sidebar.pyplot()

    # Revenus moyens
    st.sidebar.markdown("<u>Revenus moyens $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant crédits moyen
    st.sidebar.markdown("<u>Montant crédits moyen $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    # Présentation de l'identifiant client sélectionné
    st.write("Client Sélectionné :", id_client)

    # Affichage état civil
    st.header("**Etat civil et Crédit**")

    if st.checkbox("Cochez, si vous voulez des informations sur le client."):

        infos_client = identite_client()
        st.write("Statut famille :**", infos_client["NAME_FAMILY_STATUS"][0], "**")
        st.write("Nombre d'enfant(s) :**", infos_client["CNT_CHILDREN"][0], "**")
        st.write("Age client :", int(infos_client["DAYS_BIRTH"].values / -365), "ans.")

        data_age = load_age_population()
        # Set the style of plots
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(9, 9))
        # Plot the distribution of ages in years
        plt.hist(data_age, edgecolor = 'k', bins = 25)
        plt.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="red", linestyle=":")
        plt.title('Age of Client')
        plt.xlabel('Age (years)')
        plt.ylabel('Count')
        st.pyplot(fig)

        st.subheader("*Revenus*")
        #st.write("Total revenus client :", infos_client["revenus"], "$")
        st.write("Total revenus client :", infos_client["AMT_INCOME_TOTAL"][0], "$")

        data_revenus = load_revenus_population()
        # Set the style of plots
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(9, 9))
        # Plot the distribution of revenus
        plt.hist(data_revenus, edgecolor = 'k')
        plt.axvline(infos_client["AMT_INCOME_TOTAL"][0], color="red", linestyle=":")
        plt.title('Revenus du Client')
        plt.xlabel('Revenus ($ USD)')
        plt.ylabel('Count')
        st.pyplot(fig)

        st.write("Montant du crédit :", infos_client["AMT_CREDIT"][0], "$")
        st.write("Annuités crédit :", infos_client["AMT_ANNUITY"][0], "$")
        st.write("Montant du bien pour le crédit :", infos_client["AMT_GOODS_PRICE"][0], "$")
    else:
        st.markdown("<i> </i>", unsafe_allow_html=True)

    # Affichage solvabilité client
    st.header("**Analyse dossier client**")

    st.markdown("<u>Probabilité de risque de faillite du client :</u>", unsafe_allow_html=True)
    prediction = load_prediction()
    st.write(round(prediction*100, 2), "%")
    st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
    st.write(identite_client())

    # Affichage des dossiers similaires
    chk_neighbors = st.checkbox("Cochez, si vous voulez comparer avec des dossiers similaires?")

    if chk_neighbors:

        similar_id = load_neighbors()
        st.markdown("<u>Groupe des 10 clients similaires :</u>", unsafe_allow_html=True)
        st.write(similar_id)
        st.markdown("<i>Target 1 = Client en faillite</i>", unsafe_allow_html=True)
    else:
        st.markdown("<i> </i>", unsafe_allow_html=True)


@st.cache
def init_api():
    # Requête permettant de récupérer la liste des ID clients
    init_api = requests.get(URL_API + "init_model", headers={
"User-Agent" : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
})
    init_api = init_api.json()

    return "Lancement application."

@st.cache()
def load_logo():
    # Construction de la sidebar
    # Chargement du logo
    logo = Image.open("logo.png")

    return logo

@st.cache()
def load_selectbox():
    # Requête permettant de récupérer la liste des ID clients
    data_json = requests.get(URL_API + "load_data")
    data = data_json.json()

    # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i[0])

    return lst_id

@st.cache()
def load_infos_gen():

    # Requête permettant de récupérer :
    # Le nombre de lignes de crédits existants dans la base
    # Le revenus moyens des clients
    # Le montant moyen des crédits existants
    infos_gen = requests.get(URL_API + "infos_gen")
    infos_gen = infos_gen.json()

    nb_credits = infos_gen[0]
    rev_moy = infos_gen[1]
    credits_moy = infos_gen[2]

    # Requête permettant de récupérer
    # Le nombre de target dans la classe 0
    # et la classe 1
    targets = requests.get(URL_API + "disparite_target")
    targets = targets.json()


    return nb_credits, rev_moy, credits_moy, targets


def identite_client():

    # Requête permettant de récupérer les informations du client sélectionné
    infos_client = requests.get(URL_API + "infos_client", params={"id_client":id_client})
    #infos_client = infos_client.json()

    # On transforme la réponse en dictionnaire python
    infos_client = json.loads(infos_client.content.decode("utf-8"))

    # On transforme le dictionnaire en dataframe
    infos_client = pd.DataFrame.from_dict(infos_client).T

    return infos_client

@st.cache
def load_age_population():

    # Requête permettant de récupérer les âges de la
    # population pour le graphique situant le client
    data_age_json = requests.get(URL_API + "load_age_population")
    data_age = data_age_json.json()

    return data_age

@st.cache
def load_revenus_population():

    # Requête permettant de récupérer des tranches de revenus
    # de la population pour le graphique situant le client
    data_revenus_json = requests.get(URL_API + "load_revenus_population")

    data_revenus = data_revenus_json.json()

    return data_revenus

def load_prediction():

    # Requête permettant de récupérer la prédiction
    # de faillite du client sélectionné
    prediction = requests.get(URL_API + "predict", params={"id_client":id_client})
    prediction = prediction.json()

    return prediction[1]

def load_neighbors():

    # Requête permettant de récupérer les 10 dossiers client ayant des similitudes avec le client sélectionné
    neighbors = requests.get(URL_API + "load_neighbors", params={"id_client":id_client})

    # On transforme la réponse en dictionnaire python
    neighbors = json.loads(neighbors.content.decode("utf-8"))

    # On transforme le dictionnaire en dataframe
    neighbors = pd.DataFrame.from_dict(neighbors).T

    # On déplace la colonne TARGET en premier pour plus de lisibilité
    target = neighbors["TARGET"]
    neighbors.drop(labels=["TARGET"], axis=1, inplace=True)
    neighbors.insert(0, "TARGET", target)

    return neighbors

if __name__ == "__main__":
    main()
