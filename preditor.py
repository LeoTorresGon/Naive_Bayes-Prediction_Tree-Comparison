from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, cohen_kappa_score
from sklearn import tree
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
y = data['Overall_Impact'].map({'Positive':1,'Neutral':0,'Negative':-1}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treino e predição
gnb = GaussianNB()
gnb.fit(X_train, y_train)

arvore = tree.DecisionTreeClassifier()
arvore.fit(X_train, y_train)

y_pred_nb = gnb.predict(X_test)
y_pred_tree = arvore.predict(X_test)

# Métricas
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
print(f"{'MÉTRICA':<20} | {'NAIVE BAYES':<10} | {'ÁRVORE':<10}")
print("-" * 45)
print(f"{'Acurácia':<20} | {acc_nb:<10.2f} | {acc_tr:<10.2f}")
print(f"{'F1-Score (Weighted)':<20} | {f1_nb:<10.2f} | {f1_tr:<10.2f}")
print(f"{'Kappa de Cohen':<20} | {kap_nb:<10.2f} | {kap_tr:<10.2f}")
print("="*45 + "\n")

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