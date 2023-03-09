import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title='CERCOCAP---ML Project',
    page_icon=Image.open('photo/icon.jpg'),
    layout="centered",
    initial_sidebar_state="auto",
)

st.title('CERCOCAP - Prediction by LightGBM')
with st.expander('Disclaimer'):
    
    st.info('This is a demonstration of a machine learning model with cercocap.')
    st.write('This model gives a percentage prediction of the cercoporiosed of this parcel based on the geographic information and weather information related to the input parcel. (The exact format of the input data can be viewed using the sample data)')
    st.write('The model also provides key factors influencing the epidemic and a decision tree model to help the researcher make a judgement.')
    
    
acta = Image.open('photo/ACTA.png')
itb = Image.open('photo/ITB.jfif')
st.sidebar.image(acta)
st.sidebar.image(itb)
RANDOM_STATE = st.sidebar.number_input("Insert a RANDOM STATE Number", 0)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,sep=";")
elif st.button('Load example data'):
    df = pd.read_csv('output/df_zipped.gz',sep = ';',compression='gzip')
    df = df.sample(frac=0.01, random_state=RANDOM_STATE).reset_index(drop=True)
    
if df is not None:
    st.write('Here is the raw data')
    st.dataframe(df)
    df_lgb = df[["jour","valeur","zone","zone_a_risque","T_Q","HU_Q","SSI_Q","Q_Q",
                "coef_variete",
                "T_MOY","HU_MOY","SSI_MOY","Q_MOY"]]
    st.subheader('Data')

    st.write('Here is the data used for the prediction')
    st.table(df_lgb.head(3))
    
    Y_test = df_lgb["valeur"]
    X_test = pd.get_dummies(df_lgb[["jour","coef_variete","zone","zone_a_risque",
                            "T_Q","HU_Q","SSI_Q","Q_Q",
                            "T_MOY","HU_MOY","SSI_MOY","Q_MOY"]])
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
    
    lgb_model = lgb.Booster(model_file='output/lgb_model.txt')
    with st.spinner("Predicting... Please wait"):
        y_pred = lgb_model.predict(X_test)
        for i in range(len(y_pred)):
            if y_pred[i] > 100: y_pred[i] =100

    st.subheader('Prediction Result')
    #st.write('The r2 of training data is:', metrics.r2_score(Y_train, lgb_model.predict(X_train)))
    st.write('The r2 of prediction is:', metrics.r2_score(Y_test, y_pred))
    st.write('The RMSE is:' , np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
    
    st.write('The Prediction table is :')
    st.table(y_pred[1:5])
    
    
    st.write('Data Visualisation')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    baseline = pd.read_csv('output/baseline.csv',sep = ';')
    plt.plot(sorted(X_test['jour']),sorted(Y_test),label='Real')
    plt.plot(sorted(X_test['jour']),sorted(y_pred),label='Prediction')
    plt.plot(baseline['jour'],baseline['valeur'],label='Baseline', linestyle = 'dashed')
    plt.legend()
    st.pyplot()
    
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(sep=";").encode('utf-8')

    csv = convert_df(pd.DataFrame(y_pred))

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Prediction.csv',
        mime='text/csv',
    )
    
    st.subheader('Model interpretation')
    with st.expander('Feauture Importance'):
        lgb.plot_importance(lgb_model,figsize=(20,15))
        st.pyplot()
        st.write('The final decision tree of the model')
        
    with st.expander('Tree Plot'):
        dpi = plt.rcParams['figure.dpi']
        graph = lgb.create_tree_digraph(lgb_model)
        st.graphviz_chart(graph)
    
    
