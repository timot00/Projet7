# path Desktop\Documents\Data Scientist Openclassrooms\Projet 7

import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#import flask
import pickle
#from flask import Flask, render_template, request
import streamlit as st
import shap
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree
import plotly.graph_objects as go
import requests
import json

url = "http://127.0.0.1:8000/prediction"



def main() :
    

    @st.cache
    def load_data():
        description = pd.read_csv("features_description.csv", 
                                      usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')
        data = pd.read_csv('input_data_model2.zip', index_col='SK_ID_CURR', encoding ='utf-8')
        
        #data = pd.read_csv('input_data_model.csv.zip', index_col='SK_ID_CURR', encoding ='utf-8')
        data = data.drop('Unnamed: 0', 1)
        data = data.drop('index', 1)
        
        

        

        target = data.iloc[:, -1:]

        #data = data.head(500)
        #target = target.head(500)

        return data, target, description


    def load_model():
        '''loading the trained model'''
        pickle_in = open('LGBMClassifier.pkl', 'rb') 
        clf = pickle.load(pickle_in)
        return clf

    
    @st.cache(allow_output_mutation=True)
    def load_knn(data):
        knn = knn_training(data)
        return knn
    
    
    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
        round(data["AMT_INCOME_TOTAL"].mean(), 2),
        round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets



    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    
    @st.cache
    def load_age_population(data):
        data_age = -round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    
    @st.cache
    def load_income_population(data):
        df_income = pd.DataFrame(data["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income
    
    
    @st.cache
    def load_credit_population(data):
        df_credit = pd.DataFrame(data["AMT_CREDIT"])
        df_credit = df_credit.loc[df_credit['AMT_CREDIT'] < 2e6, :]
        return df_credit


    @st.cache
    def load_prediction(data, id, clf):
        X=data.iloc[:, 1:]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score
    
    def load_prediction(url, data):

    headers = {'Content-Type': 'application/json'}

    data_json = json.dumps(data)
    # print(data_json)

    response = requests.request("POST", url, headers=headers, data=data_json)
    # print(response.text)
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text)
        )
    result_client_solvable = response.json()
    # print(result_client_solvable["prediction"])
    return result_client_solvable["prediction"]




    #Loading data……
    data, target, description = load_data()
    id_client = data.index.values
    clf = load_model()
    
    
    #######################################
    # SIDEBAR
    #######################################

    #Title display
    html_temp = """
    <div style="padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard de "scoring crédit" évaluation de la solvabilité client</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    
    # Sélection du numéro client (id)
    chk_id = st.sidebar.selectbox("ID Client", id_client)

    
    
    
    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    # Champs de la barre latérale:
    st.write("ID du client:", chk_id)


    
    ## Affichage de la solvabilité client ##
    
    #
    st.header("**Score de solvabilité**")
    prediction = load_prediction(data, chk_id, clf)
    st.write("**Probabilité d'un défaut de paiement : **{:.0f} %".format(round(float(prediction)*100, 2)))


    st.markdown("<u>Données client :</u>", unsafe_allow_html=True)
    st.write(identite_client(data, chk_id))

 
    
        
if __name__ == '__main__':
    main()

