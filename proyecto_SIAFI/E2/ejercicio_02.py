"""
Sanchéz Meza Ariadna Osiris 
SIAFI | Propedéutico Técnico 2025-2
Proyecto

"""
#Ejercicio 2.
#Clasificación de frases según emojis. k-Nearest Neighbo

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer #la usamos para convertir los vectores
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #para utiliar el clasificador k-NN
from sklearn.decomposition import PCA

#Creamos un diccionario con frases randm y emojs que podrían representarlas
data = [
    {"frase": "Estoy muy feliz hoy", "emoji": "😊"},
    {"frase": "El día esta bonito", "emoji": "🌞"},
    {"frase": "Quisiera jugar D:", "emoji": "😢"},
    {"frase": "Tengo cólicos", "emoji": "😞"},
    {"frase": "Eso brilla demasiado", "emoji": "🤩"},
    {"frase": "Es muy picante", "emoji": "😡"},
    {"frase": "Por supuesto que sí", "emoji": "😌"},
    {"frase": "Se me olvido", "emoji": "😰"},
    {"frase": "Que lindoooo", "emoji": "😍"},
    {"frase": "Ni modo", "emoji": "😕"},
    {"frase": "Me dio mucha risa", "emoji": "😆"},
    {"frase": "Ay!", "emoji": "😅"},
    {"frase": "Estoy llorando de la risa", "emoji": "🤣"},
    {"frase": "Gracias!", "emoji": "😊"},
    {"frase": "Obvio", "emoji": "😇"},
    {"frase": "Fok", "emoji": "🙂"},
    {"frase": "No  puede ser", "emoji": "🙃"}
]
frases = [entry["frase"] for entry in data]
emojis = [entry["emoji"] for entry in data]

vector= TfidfVectorizer() #las frases las  convertimos a vectores  para  entrenar y predecir
X = vector.fit_transform(frases).toarray()

#aqui empezamos a dividir  en los datos prueba con un  30% y los de entremiento con un 70% es decir el 70 se utiliza para entrenar los datos y el 30 para despues del entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, emojis, test_size=0.3, random_state=42)#entrena y evalua

knn = KNeighborsClassifier(n_neighbors=3)# Utilizamos el kNN para observar el mayor numero de vecinos
knn.fit(X_train, y_train)#Cuando  lo igualamos a 3 queremos decir que tomamos los 3 vectores más cercanos
#Ahora comenzamos a predecir con x_test
y_pred = knn.predict(X_test)
#A partir de  aquí iteramos  sobre las predicciones
#comparando las reales con las de prediccion  
print("Resultados:") 
correctas = []
incorrectas = []
for i, (frase, real, pred) in enumerate(zip(frases, y_test, y_pred)):
    if real == pred:
        correctas.append((frase, real, pred))
    else:
        incorrectas.append((frase, real, pred))

print("Predicciones que son correctas:")
for frase, real, pred in correctas:
    print(f"Frase: {frase} / Dato real: {real} / Predicción: {pred}")

print("\nPredicciones que son incorrectas:")
for frase, real, pred in incorrectas:
    print(f"Frase: {frase} / Dato real: {real} / Predicción: {pred}")


fig, ax = plt.subplots()# Vamos viendo los  resultados en una gráfica para obsevar las rediccioes y elacercamiento que exise
colors = {"😊": "blue", "🌞": "yellow", "😢": "purple", "😞": "red", "🤩": "green",#Además les asigne colores porque no  se me ocurrio de otra forma identificarlos
          "😡": "orange", "😌": "cyan", "😰": "pink", "😍": "magenta", "😕": "brown",
          "😆": "lime", "😅": "gray", "🤣": "teal", "😇": "olive", "🙂": "navy",
          "🙃": "gold"}

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

for i, (frase, emoji) in enumerate(zip(frases, emojis)):#Vamos itrando para cada frase y emoji 
    ax.scatter(X_pca[i, 0], X_pca[i, 1], color=colors[emoji], #la etiqueta es el emoji
               label=emoji if emoji not in ax.get_legend_handles_labels()[1] else "")#donde aparece cada frase 

#Ahora las predicciones para los datos de prueba que se representan con una cruz
X_test_pca = pca.transform(X_test)  # Reducir dimensiones de X_test con PCA
for i, (x, y, pred) in enumerate(zip(X_test_pca[:, 0], X_test_pca[:, 1], y_pred)):
    ax.scatter(x, y, color="black", marker="x", s=100, label="Predicción" if "Predicción" not in ax.get_legend_handles_labels()[1] else "")
    ax.annotate(pred, (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=10)

ax.set_title("Frases y Emojis")#Graficamos 
ax.set_xlabel("Parte 1")
ax.set_ylabel("Parte 2")
plt.tight_layout()
plt.show()
#podemos ver que los emojis en la gráfica representan los datos reales es decir las frases
#Por otro lado las cruces representan las predicciones para los datos prueba
#Cada emoji  estarepresentado con un color y cuando vemos una cruz encima quiere decir que la predicción fue correcta
#En cambio si no hay cruz encima quiere decir que la predicción fue incorrecta
#En la terminal  podemos verque en las predicciones correctas no aparecen como tal, estas se muestran en lagráfica "fisicamente"
#Notando que en la parte de arriba se muestr el emoji de la predicción es decir la cruz
#Cuando  vemos las predicciones incorrectas son las que no aparecen  por completo en la gráfica
#Tambien se observa que en  las predicciones tienden a estar con el emoji de la frase de "ni  modo"
#Concluyo que se debe a que existe un sesgo hacia esa frase porque es similar  a  las frases de prueba o porqu es "similar" a los demás emojis
#Considero que con más frases las predcciones serán más varantes y no se sesgarán tanto