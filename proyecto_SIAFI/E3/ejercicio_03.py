"""
Sanchéz Meza Ariadna Osiris 
SIAFI | Propedéutico Técnico 2025-2
Proyecto

"""
#Ejercicio 3.
#Clasificación de videojuegos por popularidad. | Bosques aleatorios (Random Forest)

import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Cargamos el dataset de VideoGameSales 
data = pd.read_csv(r'C:\Users\andai\Escritorio\proyecto_SIAFI\E3\vgchartz-2024.csv')
data.info()
#tenemos 14 columnas con los siguientes nombres:
"""
Clasifica videojuegos como "muy populares", "moderadamente populares" o "men
populares" basándote en ventas, calificaciones y datos de usuarios activo
1. img -> Imagen del videojuego
2. title -> título del videojuego
3. console -> tipo de consola en la que se juega
4. gnre -> género del juego
5. publisher -> editor del juego (va por empresa)
6. developer -> desarrollador del juego (estudios desarrolladores de videojuegos)
7. critic_score -> puntuación de la meta crítica (sobre 10) FLOAT-----------------------------------------
8.total_sales -> Ventas mundiales de copias en millones -----------------------------------------
9. na_sales -> Ventas de copias en América del Norte en millones
10. jp_sales ->Ventas japonesas de copias en millones
11. pal_sales -> Ventas de copias en millones en Europa y África
12.other_sales -> Otras ventas de copias en millones FLOAT
13.release-date ->Fecha de lanzamiento del juego, la fecha la toma como objeto por el uso de / entre numeros
14.last_update ->Fecha en la que se actualizaron los datos por última vez, igual aqui es tipo objeto
"""
#Vamos a limpiar los datos  
#Nos damos  cuenta que las dos columnas release_date y last_update tinen muchos valores nulos por lo que las eliminaremos
#Lo puse aquí arriba porque me di cuenta que más adelante no lo necesiaríamos 
data = data.drop(columns=['release_date', 'last_update'])
# Ahora sí podemos tomar las  columnas de tipo numérico
num_cols=data.select_dtypes(include='number').columns 
#Ahora de forma numerica 
outliers = {}
for col in num_cols: 
    Q1 = np.percentile(data[col], 25) 
    Q3 = np.percentile(data[col], 75)
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[col] = (data[col] > upper_bound).sum() + (data[col] < lower_bound).sum()

print("\n################# OUTLIERS de columnas númericas ################")
for col, count in outliers.items():#lo usamos para mostrar en forma de lista
    print(f"- {col}: {count} outliers")
#Nos  indica que en las  columnasnúmericas no hay outliers

#Vamos a ver si existen valores nulos
print("\nIdentificamos valores nulos:")
data.isna().sum()/len(data)
num_null = data.isna().sum()
print(num_null[num_null > 0]) 
"""
las columnas con valores nulos son las siguientes:
developer          17,critic_score    57338, total_sales     45094,  na_sales        51379, jp_sales        57290
pal_sales       51192, other_sales     48888, release_date     7051, last_update     46137
"""
"""
#==================================================================================================================================
#Decidí gráficar antes de quitar los valores nulos para que se pueda observar un antes  u despues
for col in num_cols:  
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)  
    plt.title(f'Con valores nulos columna: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
"""
#Manejar  los valores nulos
#Como no quise eliminar los datos y despues llenar lo que hice fue lo sifuiente:
#llenamos los valores nulos de las columnas numericas con la mediana 
numeric_cols = ['critic_score', 'total_sales', 'na_sales', 'jp_sales', 'pal_sales', 'other_sales']
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())
print("\nValores nulos después de llenar:")
print(data[numeric_cols].isna().sum())

#=========================================================================================================
"""
#Gráficamos (Ya sin valores nulos)
#Ahora vamos a ver las gráficas de cada columna númerica para decidir si se modifican sus valores, según sea el caso
for col in num_cols:  # num_cols ya contiene las columnas numéricas
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)  #El KDE dibuja una curva de densidad para ver mejor la distribución de cada columna
    plt.title(f'Sin valores nulos, columna: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
"""
#Ahora con las variables categóricas
# Identificar las columnas de tipo objeto 
obj_cols = data.select_dtypes(include='object').columns
print("\nNombres de columnas tipo objeto:\n") #mostramos las columnas tipo objeto
for col in obj_cols:
    print(f"- {col}")
#que columnas tienen valores nulos
print("\nColumnas tipo objeto con valores nulos:")
for col in obj_cols: #el for lo utilizamos para mostrar solo las columas que contienen valores nulos, para despues llenarlo
    nulos_antes = data[col].isna().sum()
    if nulos_antes > 0:  # Las columnas solo con valores nulos
        print(f"Columna: {col}")
        print(f"Nulos: {nulos_antes}")
#Eliminamos filas con valores nulos en la columna developer because son solo 17 filas por lo tanto no afecta en 
#el analisis de los datos porque si la lleno con otra letra sería insignificante considerarla
data = data.dropna(subset=['developer'])
print(f"\nValores nulos en developer: {data['developer'].isna().sum()}")

#=========================================================================================================

#Vamos a ver la varianza para observar cuan dipersas estan cada columna
#Obtenemos la varianza de las columnas númericas   
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
print("\nVarianza de las columnas numéricas:\n")
variance = data[num_cols].var()
print(variance.to_string())

#Ahora con columnas categoricas
print("\nVarianza de las columnas categoricas:\n")
cat_cols=data.select_dtypes(include='O').columns 
for c in cat_cols:
  print('\nColumna :',c)
  print(data[c].value_counts()) 
#Podemos   observar que estamos manejando una varianza muy amplia para todas las columnas 
correlation_matrix = data[num_cols].corr()
print(correlation_matrix)

#Normalizamos para que se pueda comparar las columnas
scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])
print(data[num_cols].head())

#Vemos el mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()
#observaos que las columnas como total_sales, na_sales, jp_sales, pal_sales y other_sales tienen una correlación alta


#Ahora necesitamos hacer uso de los persentiles para  clasificar los videojuegos como
#muy populares, moderadamente populares o menos populares de 'total_sales'

# Lo hare de 33 y 66 respectivamente
# Calcular los percentiles 33 y 66 para la columna 'total_sales'
percentile_33 = data['total_sales'].quantile(0.33)
percentile_66 = data['total_sales'].quantile(0.66)

# Popularity será una nueva columna para meter las clasificaciones
def classify_popularity(sales):
    if sales > percentile_66:#A partir de aqui usamos return para que retorne un número y no itere en la salida como sería con print	
        return 'muy populares'
    elif sales > percentile_33:
        return 'moderadamente populares'
    else:
        return 'menos populares'
# Ahora si se van mostrando dependiendo  a que pertenezca cada categoría
data['popularity'] = data['total_sales'].apply(classify_popularity)
categories = ['muy populares', 'moderadamente populares', 'menos populares']
data['popularity'] = pd.Categorical(data['popularity'], categories=categories, ordered=True)
print("\nGráfica de popularidad:")
print(data['popularity'].value_counts())

# Graficar la distribución para visualizar mejor las categorías
plt.figure(figsize=(8, 4))
sns.countplot(data=data, x='popularity', hue='popularity', palette='viridis', order=categories, legend=False)
plt.title('Distribución de Videojuegos por Popularidad')
plt.xlabel('Popularidad')
plt.ylabel('Cantidad de Videojuegos')
plt.show()
# y senota que no existen videojuego moderadamente populares
#=========================================================================================================

#procedemos  con el entrenamiendo  
#Con ayuda del mapa de calor  decidí que trabajar con las columnas altamente relacionadas sería lo mejor
#Pero eso no es todo, es decir, cuando un producto es altamente popular en una región es probable que también lo sea en otra
#ya sea porque comparten gustos, por globalización o por que en esos paises es fácil su adquisisión 
#es por eso que decidí trabajar con las columnas de region como:na_sales -> Ventas de copias en América del Norte 
#jp_sales ->Ventas japonesas de copias y pal_sales -> Ventas de copias en Europa y África ya que es más probable que aumenten las ventas y 
#es más especifico que por ejemplo escoger other_sales -> Otras ventas de copias porque no es tan específico
features = ['na_sales', 'jp_sales', 'pal_sales']  # Usar solo las ventas regionales como características
target = 'popularity'  # Columna objetivo
X = data[features]
y = data[target]

#vamos a asignar los datos de entrenamiento y de prueba (porque estaremos utilizando el modelo random forest
# Dividir los datos en entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42) # Implementamos el modelo random Forest
#Entrenamos el modelo
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluamos el modelo para ver que tan exacto es el modelo
exactitud = accuracy_score(y_test, y_pred)
print(f"Nivel de exactitud: {exactitud:.4f}")
#en los resutados  arroja  que tiene 0.9912 de exactitud lo que quiere decir que la  mayoria de sus predicciones son correctas
#por lo tanto el modelo funciona bien



conf_matrix = confusion_matrix(y_test, y_pred)  # Matriz de confusión del modelo que basicamente estamos viendo si clasifica bien
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=data['popularity'].cat.categories, yticklabels=data['popularity'].cat.categories)
plt.title("Matriz de Confusión del Modelo")
plt.xlabel("Predicción :O")
plt.ylabel("Realidad :)")
plt.show()
#en tabla de salida podemos notar que la mayoria de  las prediccionesesta con 0.----
#Lo que funciona como una "reafirmación" de la exactitud del modelo
#además de que lo muestra para cada correlación de columna

importances = rf_model.feature_importances_ # Obteniendo las importancias de cada característica
feature_names = X.columns
#Para esto cree otra dataframe para que sea independiente y se pueda  ver mejor 
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6)) #y graficamos 
sns.barplot(x="importance", y="feature", data=feature_importances_df)
plt.title("Importancia de Características Según el Modelo")
plt.show()







