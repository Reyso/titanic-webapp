import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import joblib
import time


# Carregando o modelo
model = joblib.load('clf-best.pickle')

# Carregando dataset
data = pd.read_csv("train.csv")



# função main

def main():
    st.title('Você sobreviveria ao titanic? :ship:')
    st.subheader('Faça seu Teste!')
    st.write('Preencha todos os campos abaixo e veja qual seria sua chance de SOBREVIVER AO TITANIC')
    
    
    #side bar
    rad = st.sidebar.radio('Navigaton',['Home','About me'])
    if rad == 'About me':
        st.sidebar.header("I'm a data scientist in evolution 🚀")
        st.sidebar.write('🟢 -I’m currently learning Scikit learn, Pyspark , Flask API, Hadoop, Amazon Web Services')
        st.sidebar.write('🟡 -I’m looking to collaborate on data science projetcs an AI')
        st.sidebar.write('🔴 -All of my projects are available at https://github.com/Reyso')
        st.sidebar.write('🔵 -How to reach me reyso.ct@gmail.com')
        
    st.sidebar.subheader('Connect with me: https://github.com/Reyso/')
    
    #Inputs
    passenger_name = st.text_input('Insira seu nome','Rose')
    passenger_class = st.select_slider("Classe do passageiro", [1,2,3], 2)
    sex = st.selectbox('Sexo', ['female', 'male'])
    age = int(st.slider('Idade:', 0, 100, 20))
    sibsp = st.slider('Quantidade de irmãos/conjudes abordo:',0,10,2)
    parch = st.slider('Quantidade de Pais/Crianças abordo:',0,10,2)
    fare = st.number_input("Tarifa")
    embarked = st.selectbox("Embarque: ", data["Embarked"].unique())
    st.write(' C = Cherbourg, Q = Queenstown, S = Southampton')

    
        # Criando um dicionário para os dados de entrada
    input_data = {
        "Pclass": [passenger_class],
        "Sex": [sex],        
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]

    }
    
    # transformando dados de entrada em um dataframe
    passager = pd.DataFrame(input_data)

    
    
    
    st.write('Suas informações: ')
    st.write(passager)
    
    
    
        # Utilizando o model para faze a prvisão
    if st.button("Predict 👈"):
        prediction = model.predict(passager)
        if prediction[0] == 1:
            
            with st.spinner('Só um momento...'):
                time.sleep(5)
            st.balloons()
            st.success('VOCÊ SOBREVIVEU🎉! Sua história poderia virar um filme!')
            
            
    
        else:
            
            with st.spinner('Só um momento...'):
                st.write(passenger_name)
                time.sleep(5)
            st.snow()
            st.error('morremo 😰, talvez tivesse lugar pra você em alguma prancha')
    
    


# Run  no app
if __name__== "__main__":
    main()


#Pclass	Sex	Age	SibSp	Parch	Fare	Embarked

#Passager ID - 123456
#PClass - 1,2,3
#Name - 'Jack'
#Sex - 'masculino','feminino'
#Age - 0,100
#Sisp - 0,10 - Numero de irmãos ou esposos(a)
#Parch - 0,2 - numero de pais ou crianças abordo
#ticket - 12345
#fare - 0,100 - tarifa - preço da passagem
# Cabin - c52
# Embarked - S,C,Q  - C = Cherbourg, Q = Queenstown, S = Southampton
