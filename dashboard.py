import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import requests
import numpy as np
import json

#conda deactivate
#conda activate env7
#streamlit run "C:\Users\Me\Documents\1- DATA SCIENCE\Projet7\DASHBOARD\dashboard.py"
#run dans le terminal: streamlit run dashboard.py 
#.\projet7\Scripts\activate 
#env7

# chargement des données
df = pd.read_csv('bin/df_dash.csv', encoding='utf-8')
df_no_target = df.drop(['TARGET'], axis=1)

# To set a webpage title, header and subtitle
st.set_page_config(page_title="Scoring Bancaire", layout='wide')
st.header("Application de détection de faillite bancaire")

PAGES = [
	    "Tableau clientèle",
	    "Comparaison clientèle",
		"Visualisation score"

	]

st.sidebar.image('bin/logopretadepenser.JPG')
st.sidebar.title('Menu')
selection = st.sidebar.radio("Utilisez le menu pour naviguer entre les différentes pages.", PAGES)


def filtrecolonne(df):
    # Colonnes à filtrer
    tofiltercolselect = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS']
    tofiltercolinter = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL']

    # Créer des colonnes pour les filtres de sélection
    select_columns = st.columns(len(tofiltercolselect))
    for i, col in enumerate(tofiltercolselect):
        filter_value = select_columns[i].selectbox(f"Filtrer par {col}", ['all'] + list(df[col].unique()))
        if filter_value != 'all':
            df = df[df[col] == filter_value]

    # Créer des colonnes pour les filtres d'intervalle
    inter_columns = st.columns(len(tofiltercolinter))
    for i, col in enumerate(tofiltercolinter):
        filter_min = inter_columns[i].number_input(f"Minimum pour {col}", min_value=df[col].min(), max_value=df[col].max(), value=df[col].min())
        filter_max = inter_columns[i].number_input(f"Maximum pour {col}", min_value=df[col].min(), max_value=df[col].max(), value=df[col].max())
        df = df[(df[col] >= filter_min) & (df[col] <= filter_max)]

    # Afficher le DataFrame filtré
    len_df = len(df)
    st.write(f"Nombre de correspondances trouvées: {len_df}")
    st.write(df)
   
def infoclient(selected_client):
    st.markdown("<h4 style='font-weight: bold;'>Informations sur le client sélectionné :</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.write("ID client : " + str(selected_client['SK_ID_CURR'].item()))
    col1.write("Genre : " + selected_client['CODE_GENDER'].item())
    col1.write("Type de contrat : " + selected_client['NAME_CONTRACT_TYPE'].item())
    col1.write("Nombre d'enfant(s) : " + str(selected_client['CNT_CHILDREN'].item()))
    col2.write("Revenu total : " + str(selected_client['AMT_INCOME_TOTAL'].item()))
    col2.write("Type de revenu : " + selected_client['NAME_INCOME_TYPE'].item())
    col2.write("Type d'éducation : " + selected_client['NAME_EDUCATION_TYPE'].item())
    col2.write("Statut familial : " + selected_client['NAME_FAMILY_STATUS'].item())
    col2.write("Type de logement : " + selected_client['NAME_HOUSING_TYPE'].item())




from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

def comparaison_client():
    # Liste déroulante pour sélectionner le numéro du client
    liste_id = df['SK_ID_CURR'].tolist()
    id_input = st.selectbox("Sélectionnez le numéro du client", df['SK_ID_CURR'])
    if id_input == '':
        st.write("Pas de client sélectionné")
    elif int(id_input) in liste_id:
        selected_client = df[df['SK_ID_CURR'] == int(id_input)]
        st.write(selected_client.drop(columns=['TARGET'],axis=1))
        infoclient(selected_client)
        plt.figure(figsize=(16, 12))
        st.markdown("<h3 style='font-weight: bold;'>Graphiques de comparaison avec les autres clients :</h3>", unsafe_allow_html=True)
        # selection nb knn
        min_neighbors = 100
        max_neighbors = len(df)
        default_neighbors = 15000
        inter_columnss = st.columns(3)
        num_samples = inter_columnss[0].slider("Sélection du nombre de clients les plus proches pour la comparaison", min_value=min_neighbors, max_value=max_neighbors,
                                value=default_neighbors, step=100, help="Nombre de plus proches voisins")

        # encoder pr knn
        text_cols = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'CODE_GENDER']
        encoded_data = df[text_cols].copy()
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(encoded_data)
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(text_cols))

        for i, col in enumerate(text_cols):
            # dataframe voisins
            knn = NearestNeighbors(n_neighbors=num_samples)
            knn.fit(encoded_data)
            distances, indices = knn.kneighbors(encoded_data.loc[selected_client.index])

            neighbors = df.iloc[indices.flatten()]

            plt.subplot(2, 3, i+1)
            sns.countplot(data=df, x=col, hue='TARGET', palette={0: 'lightblue', 1: 'salmon'})
            plt.xlabel(col)
            plt.xticks(rotation=45,size=8)
            plt.ylabel('Fréquence')
            plt.legend()

            # Encadrer
            selected_category = selected_client[col].item()
            category_index = df[col][df[col] == selected_category].index[0]
            plt.axvline(x=category_index -0.5 , color='black', linestyle='--', linewidth=2)
            plt.axvline(x=category_index + 0.5, color='black', linestyle='--', linewidth=2)

        #  KDE pour 'CNT_CHILDREN' et 'EXT_SOURCE_3'
        plt.subplot(2, 3, 5)
        colors = sns.color_palette(['lightblue', 'salmon'])
        sns.kdeplot(data=neighbors[neighbors['TARGET'] == 0], x='CNT_CHILDREN', label='0',color=colors[0])
        sns.kdeplot(data=neighbors[neighbors['TARGET'] == 1], x='CNT_CHILDREN', label='1',color=colors[1])
        plt.axvline(x=selected_client['CNT_CHILDREN'].item(), color='black', linestyle='--', label='Client sélectionné')
        plt.xlabel('CNT_CHILDREN')
        plt.ylabel('Densité')
        plt.legend()

        plt.subplot(2, 3, 6)
        sns.kdeplot(data=neighbors[neighbors['TARGET'] == 0], x='EXT_SOURCE_3', label='0',color=colors[0])
        sns.kdeplot(data=neighbors[neighbors['TARGET'] == 1], x='EXT_SOURCE_3', label='1',color=colors[1])
        plt.axvline(x=selected_client['EXT_SOURCE_3'].item(), color='black', linestyle='--', label='Client sélectionné')
        plt.xlabel('EXT_SOURCE_3')
        plt.ylabel('Densité')
        plt.legend()
        plt.subplots_adjust(hspace=0.5) 
        st.pyplot(plt)

def visualisation_score():
    # Liste déroulante pour sélectionner le numéro du client
    liste_id = df_no_target['SK_ID_CURR'].tolist()
    id_input = st.selectbox("Sélectionnez le numéro du client", df_no_target['SK_ID_CURR'])
    if id_input == '': #lorsque rien n'a été saisi
        st.write("Pas de client sélectionné")

    elif (int(id_input) in liste_id): #quand un identifiant correct a été saisi on appelle l'API
        selected_client = df[df['SK_ID_CURR']==int(id_input)]
        st.write(selected_client.drop(['TARGET'], axis=1))
        infoclient(selected_client)
        st.markdown("<h4 style='font-weight: bold;'>Prédiction de Faillite Bancaire :</h3>", unsafe_allow_html=True)
        selected_client_dict = selected_client.to_dict(orient='records')
        json_data = json.dumps(selected_client_dict[0])
        #response = requests.post('http://localhost:8000/prediction/', data=json_data)
        response = requests.post('https://api-pred-banc-7-c37e4c73e633.herokuapp.com/prediction/', data=json_data)
        response_data = response.json()
        response_data = json.loads(response_data)
        # Récupération de la prédiction et d'autres résultats
        prediction = response_data["prediction"]
        score = response_data["score"]
        #"X_shap": X_shap.to_json(), "shap_values_selected": shap_values_selected.tolist(), "expected_value": expected_value}
        X_shap = pd.read_json(response_data["X_shap"])
        shap_values = response_data["shap_values_selected"]
        shap_values =shap_values[0]
        shap_values = np.array(shap_values)
        expected_value = response_data["expected_value"]
        if prediction == 1:
            st.markdown("<h5 style='color: red; font-weight: bold;'>Attention client à risque !</h4>", unsafe_allow_html=True)
        else:
            st.markdown("<h5 style='color: green; font-weight: bold;'>Le client ne semble pas à risque</h4>", unsafe_allow_html=True)
        prediction = "Non" if prediction == 0 else "Oui"
        st.write("Prédiction de faillite bancaire :", prediction)
        st.write("Score de faillite bancaire (seuil 0,53) : ", str(round(score,4)))
        #graphique shap
        plt.switch_backend('Agg')
        updated_X_shap = X_shap.copy()
        for col in updated_X_shap.columns:
            if col in selected_client.columns:
                value = selected_client[col].values[0]
                updated_X_shap[col] = value

        shap.force_plot(expected_value,shap_values,updated_X_shap,matplotlib=True, show=True)
        fig = plt.gcf() 
        st.pyplot(fig)

if selection=="Tableau clientèle":
    st.subheader("Parcourir les données clients")
    filtrecolonne(df_no_target)
if selection=="Comparaison clientèle":
    st.subheader("Comparaison clientèle")
    # Code pour la comparaison du client
    st.write("Comparaison du client avec les autres clients")
    comparaison_client()
    
if selection=="Visualisation score":
    st.subheader("Visualisation des scores de prédiction")
    st.markdown('Veuillez sélectionner un numéro de demande de prêt')
    visualisation_score()




