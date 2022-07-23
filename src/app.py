# Ahora vamos a cargar lo que precisamos para ejecutar el pipeline

import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import seaborn as sns
import pickle

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')

X = df_raw.drop(columns=['Outcome'])
y = df_raw['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=121)

# Elegimos el que mejor nos dio en esta caso hay dos iguales 3 y 8, vamos a seleccionar el 3

clf = DecisionTreeClassifier(criterion='entropy',
                             min_samples_split=20,
                             min_samples_leaf=5,
                             random_state=0, max_depth=3)

clf.fit(X_train, y_train)
print('Accuracy:',clf.score(X_test, y_test))

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
target_names = ['Negativo', 'Positivo']

print("Datos para el Train")
print(classification_report(y_train, y_train_pred, target_names=target_names))
print("="*70)
print()
print("Datos para el test")
print(classification_report(y_test, y_test_pred, target_names=target_names))

pickle.dump(clf, open('../models/decision_tree.pkl', 'wb'))
print("Se ha guardado correctamente el archivo decision_tree en la carpeta models")
print("Fin del proceso del pipeline")
