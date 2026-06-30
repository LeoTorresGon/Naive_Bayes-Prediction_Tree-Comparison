from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, cohen_kappa_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# Carregar e processar os dados
data = pd.read_csv("Social_media_impact_on_life.csv")
data = data.drop_duplicates().dropna()
data = data.drop(columns=["Mental_Health_Score", "Student_ID"])
data['Gender'] = data['Gender'].map({'Male': 0,'Female': 1})
data['Academic_Level'] = data['Academic_Level'].map({'Undergraduate': 0,'Graduate': 1,'High School': 2})
data['Affects_Academic_Performance'] = data['Affects_Academic_Performance'].map({'No': 0,'Yes': 1})
data['Country'],_=pd.factorize(data.Country)
data['Most_Used_Platform'],_=pd.factorize(data.Most_Used_Platform)

X = data.drop(columns="Overall_Impact").values
y = data['Overall_Impact'].map({'Positive':2,'Neutral':1,'Negative':0}).values # Mudar no artigo: 1 0 -1 -> 2 1 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treino e predição
gnb = GaussianNB()
gnb.fit(X_train, y_train)

arvore = tree.DecisionTreeClassifier()
arvore.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])

y_pred_nb = gnb.predict(X_test)
y_pred_tree = arvore.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_xgb = clf.predict(X_test)

# Métricas
print("\n" + "=" * 45)
print(f"Distribuição: {data['Overall_Impact'].value_counts()}")

def imprimir_resumo(modelo_nome, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"--- {modelo_nome.upper()} ---")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    return acc, f1, kappa

print("\n" + "="*45)
acc_nb, f1_nb, kap_nb = imprimir_resumo("Naive Bayes", y_test, y_pred_nb)
print("\n" + "="*45)
acc_tr, f1_tr, kap_tr = imprimir_resumo("Decision Tree", y_test, y_pred_tree)
print("\n" + "="*45)
acc_rf, f1_rf, kap_rf = imprimir_resumo("Random Forest", y_test, y_pred_rf)
print("\n" + "="*45)
acc_xgb, f1_xgb, kap_xgb = imprimir_resumo("XGBoost", y_test, y_pred_xgb)


import pandas as pd

import pandas as pd

def imprimir_tabela_pandas(dados_modelos):
    df = pd.DataFrame(dados_modelos).map(lambda x: f"{x:.2f}")
    
    df.index.name = 'MÉTRICA'
    
    print("\n" + "="*55)
    print(df.to_string())
    print("="*55 + "\n")

# --- Como usar ---
resultados = {
    'NAIVE BAYES': {'Acurácia': acc_nb, 'F1-Score (Weighted)': f1_nb, 'Kappa de Cohen': kap_nb},
    'ÁRVORE':      {'Acurácia': acc_nb, 'F1-Score (Weighted)': f1_tr, 'Kappa de Cohen': kap_tr},
    'RANDOM FOREST': {'Acurácia': acc_rf, 'F1-Score (Weighted)': f1_rf, 'Kappa de Cohen': kap_rf},
    'XGBOOST': {'Acurácia': acc_xgb, 'F1-Score (Weighted)': f1_xgb, 'Kappa de Cohen': kap_xgb},

}

imprimir_tabela_pandas(resultados)

imprimir_tabela_pandas(resultados)

print("Matriz de Confusão (Árvore de Decisão):")
print(confusion_matrix(y_test, y_pred_tree))

# Plot da árvore de decisão
features = data.drop(columns="Overall_Impact").columns
classes = ['Negative', 'Neutral', 'Positive']

plt.figure(figsize=(85, 35), dpi=300)

tree.plot_tree(arvore, 
               feature_names=features,  
               class_names=classes,
               filled=True,             
               rounded=True,            
               fontsize=8,
               precision=2)

plt.title("Árvore de Decisão:")
plt.savefig("arvore_de_decisão.png", bbox_inches='tight')