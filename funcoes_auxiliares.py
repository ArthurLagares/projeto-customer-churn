import pandas as pd
import numpy as np
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def gerar_modelos(X_treino, y_treino, X_teste, y_teste): 
    modelos = {
        'Árvore simples': DecisionTreeClassifier(),
        'Floresta Aleatória' : RandomForestClassifier(),
        'Naive Bayes' : GaussianNB()
    }


    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        print('Modelo: ', nome, '\n')
        print('Acurácia (treino): ', modelo.score(X_treino, y_treino))

        y_prev = modelo.predict(X_teste)

        print('Acurácia (teste): ', accuracy_score(y_teste, y_prev),'\n\n')



# balanceamento Undersampling

def undersampling (X_treino, y_treino):
    base_treino = pd.concat([X_treino, y_treino], axis = 1)
    indices = base_treino['churn'] == 1


    base_treino_pos = base_treino.loc[indices]
    base_treino_neg = base_treino.loc[~indices]

    valor_minimo = np.min([len(base_treino_pos), len(base_treino_neg)])


    # filtro por meio de uma amostra aleatória

    base_treino_neg = base_treino_neg.sample(n = valor_minimo, random_state= 0)
    base_treino_pos = base_treino_pos.sample(n = valor_minimo, random_state= 0)

    base_treino = pd.concat([base_treino_neg, base_treino_pos], axis = 0 , ignore_index= True)

    X = base_treino.drop('churn', axis = 1)
    y = base_treino['churn']
    return X, y



def avaliar_modelo (modelo,X_treino, y_treino, X_teste , y_teste ):

    modelo.fit(X_treino, y_treino)
    y_prev = modelo.predict(X_teste)

    cnf_matrix = confusion_matrix(y_teste, y_prev)
    
    cnf_table = pd.DataFrame(data=cnf_matrix, index=["Churn=Não", "Churn=Sim"], columns=["Churn(prev)=Não", "Prev. Churn(prev)=Sim"])
    print(cnf_table, '\n\n')

    print(classification_report(y_teste, y_prev), '\n\n')