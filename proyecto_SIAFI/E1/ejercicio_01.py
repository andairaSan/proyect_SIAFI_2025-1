"""
Sanchéz Meza Ariadna Osiris 
SIAFI | Propedéutico Técnico 2025-2
Proyecto

"""
#Ejercicio 1_.
#Predicción de streams de canciones de Spotify. Regresión lineal.

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Cargar los datos desde el archivo CSV

data = pd.read_csv(r"C:\Users\andai\Escritorio\proyecto_SIAFI\E1\spotify-2023.csv", encoding="latin1") 

#Lo primero que debemos hacer es saber que tipo de información contiene el DataFrame para eso utilizamos lo siguiente 

data.head() #Aqui observamos enforma de tabla las columnas y los datos de cada una 
print(data.head) #Pero  vemos que no es posible tener el nombre de todas las columnas 

data.info() #Por lo tanto utilizamos data.info ya que nos da los datos tabulares (todas las columnas)
print(data.info) # y además el tipo de información que maneja cada columna así como el tamaño de la tabla
print("\n")

print("////////////////////////////////////////////////////////////////////////////////////////////////////////////")
print("Información de streams ->\n") # Para revisar si hay caracteres que interrumpan la limpieza de datos
print(data['streams'].unique()) #Nos damos cuenta de que existen caracteres en un elemento de la columna

print("Información de in_deezer_playlists ->\n") # En esta parte encontramos números dentro de los números
print(data['in_deezer_playlists'].unique()) # Por lo que reemplazamos comas

print("Información de in_shazam_charts ->\n")
print(data['in_shazam_charts'].unique())  #para está parte es lo mismo que para in dezeer playlist

# Para que podamos hacer la conversión primero tenemos que 
# reemplazar los valores nulos dentro de in_shazam_charts
data['in_shazam_charts'] = data['in_shazam_charts'].fillna('0')

# Aquí remplazamos convertimos a string para poder reemplazar con comas
data['in_shazam_charts'] = data['in_shazam_charts'].astype(str).str.replace(',', '', regex=True)
data['in_shazam_charts'] = data['in_shazam_charts'].astype(int)

data['streams'] = data['streams'].astype(str)  
data = data[data['streams'].str.isdigit()]
data['streams'] = data['streams'].astype(int)  # Hacemos la conversión de objero a entero

# Reemplazamos por comas a in_deezer_playlists e in_shazam_charts
data['in_deezer_playlists'] = data['in_deezer_playlists'].astype(str).str.replace(',', '', regex=True).astype(int)
data['in_shazam_charts'] = data['in_shazam_charts'].astype(str).str.replace(',', '', regex=True).astype(int)

print("\nInformación de streams después de filtrar ->\n") #Volvemos a revisar si quedaron bien los datos
print(data['streams'].unique())

print("\nInformación después de procesar in_deezer_playlists ->\n")
print(data['in_deezer_playlists'].unique())

print("\nInformación después de procesar in_shazam_charts ->\n")
print(data['in_shazam_charts'].unique())

#Se realizo de correctamente la transformación y llenado de datos
data.info()
print(data.info)



print("===========================================================================================================0")
# Ahora sí podemos tomar las  columnas de tipo numérico
num_cols=data.select_dtypes(include='number').columns 
for col in num_cols: # y graficamos cada columna en diagramas de caja para conocer los outliers
  plt.figure(figsize=(8,2))
  sns.boxplot(data=data[num_cols], x=col)
  plt.title(f'Columna:{col}')
  plt.show()

#en cada gráfica podemos observar que existen outliers con exepción de released_month,
#released_day, valence y acousticness
#para comporbar esto lo hremos ahora de forma númerica con IQR
outliers = {}
for col in num_cols: 
    Q1 = np.percentile(data[col], 25) 
    Q3 = np.percentile(data[col], 75)
    IQR = Q3 - Q1  
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[col] = (data[col] > upper_bound).sum() + (data[col] < lower_bound).sum()

print("### OUTLIERS de forma númerica\n####")
for col, count in outliers.items():#lo usamos para mostrar en forma de lista
    print(f"- {col}: {count} outliers")
#por lo tanto podemos comprobar que efectivamente se cumple 0 datos se aislan del grupo



print("##############################################################################################################")
print("\nIdentificamos valores nulos:")
data.isna().sum()/len(data)
num_null = data.isna().sum()
print(num_null[num_null > 0])  # Aquí podemos ver solo las columnas con valores nulos
#y vemos que no son muchos en la escala que estamos manjando que son 953 datos

#Eliminamos los valores nulos y los mostramos ya sin ellos para tener más precisión 
key_len = len(data.key.dropna())
print(f"\nNúmero de valores no nulos en la columna 'key': {key_len}")
shazams_len=len(data.in_shazam_charts.dropna())
print(f"\nNúmero de valores no nulos en la columna 'in_shazam_charts': {shazams_len}")

#Ahora vamos a ver las gráficas de cada columna númerica para decidir si se modifican sus valores, según sea el caso
for col in num_cols:  # num_cols ya contiene las columnas numéricas
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)  #El KDE dibuja una curva de densidad para ver mejor la distribución de cada columna
    plt.title(f'Columna: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
#se puede apreciar que casi todas tienen una distribución normal ya que la mayoria de datos se almacenan por debajode la campanade Gauss
#pero  carecen de poca simetria 



print("###############################################################################################################################")
#Para evitar los outliers y tener mejor precisión en los calculos utilizamos la mediana y así rellenar los huecos faltantes  
impute = SimpleImputer(strategy='median') #sacams la mediana
data[num_cols] = impute.fit_transform(data[num_cols]) #imputamos las columnas númericas
print("### Datos después de imputar con mediana ###")
print(data[num_cols].isna().sum())  # Verificamos que ya no haya valores nulos
print("\nMediana:")
print(data[num_cols].median())  # Mediana de las columnas con imputación



print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
#ahora veremos los valores nulos de las columnas tipo objeto y rellenaremos
#Para: track_name, artist(s)_name, streams,in_deezer_playlists,in_deezer_charts, in_shazam_charts, key, mode
# Identificar las columnas de tipo objeto 
obj_cols = data.select_dtypes(include='object').columns
print("\nNombres de columnas tipo objeto:\n") #mostramos las columnas tipo objeto
for col in obj_cols:
    print(f"- {col}")

print("\nColumnas tipo objeto con valores nulos:\n")
for col in obj_cols: #el for lo utilizamos para mostrar solo las columas que contienen valores nulos, para despues llenarlo
    nulos_antes = data[col].isna().sum()
    if nulos_antes > 0:  # Las columnas solo con valores nulos
        print(f"Columna: {col}")
        print(f"Nulos: {nulos_antes}")
        
        # Mostramos los valores únicos antes de rellenar
        print(f"Valores únicos en inicio:\n{data[col].value_counts()}\n")
        # Creamos un imputador para rellenar con S
        impute = SimpleImputer(strategy='constant', fill_value='S')
    
        # Aplicamos la imputación a la columna
        salida = impute.fit_transform(data[[col]])
        data[col] = pd.DataFrame(salida, columns=[col])
    
        # Mostramos el número de valores nulos después de rellenar
        nulos_despues = data[col].isna().sum()
        print(f"- Nulos después: {nulos_despues}")
    
        # Mostramos los valores únicos después de rellenar
        print(f"- Valores únicos después:\n{data[col].value_counts()}\n")



print("####################################################################################################################")
#Obtenemos la varianza de las columnas númericas   
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
print("\nVarianza de las columnas numéricas:\n")
variance = data[num_cols].var()
print(variance.to_string())

#Ahora con columnas categoricas
print("\nVarianza de las columnas categoricas:\n")
cat_cols=data.select_dtypes(include='O').columns # .select_dtypes(include='O') -> seleccionamos columnas categóricas
#Existe mucha varianza entre las cantidades por lo que tenemos que normalizar
for c in cat_cols:
  print('\nColumna :',c)
  print(data[c].value_counts()) # .value_counts() -> Número de observaciones por cada categoría.
# Para artist_name, track_name existen muchas categorias por lo que no nos sirve, key tiene muchas categorias; no sirve,
#para mode existen solo dos categorias, sirve 
 
print("#################################################################################################################")

# Estandarizar las características numéricas para que tengan una media de 0 y desviación estándar de 1
# para que los datos tengan el menor sesgo posible
scaler=StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])
print("### Datos normalizados ###")
print(data[num_cols].head())  
#Podemos observar  que los datos ya se encuentran normalizados 

print("#################################################################################################################")
#Procedemos a calcula la correlación entre las columnas númericas
#Para ubicar cuales son las que tinen una relacion fuerte con streams
#Para poder realizar la predicción de los streams
correlation_matrix = data[num_cols].corr()
print(correlation_matrix) 

print("Mapas de calor")#Vemos el mapa de calor
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

#Con mapa de calor nos dimos cuenta que las columnas que tinen una relación bastante fuerte son: in_spotify_playlists e in_apple_playlists
#por lo que las utilizaremos para hacer la regresión lineal y poderr predecir los streams
plt.title("Matriz de Correlación")#Titulo del mapa
plt.show()
print("")

print("######################################################################################")
#Ahora que sabemos con que columnas vamos a trabajar procedemos con la regresión lineal
print("Regresión lineal")
reg = LinearRegression()

# Lo entrenamos con: X (datos a entrenar -> in_spotify_playlists e in_apple_playlists) , Y (datos a predecir -> streams)
reg.fit(data[['in_spotify_playlists', 'in_apple_playlists']], data['streams'])
print(f"Coeficientes de la regresión: {reg.coef_}")  # Coeficientes de la regresión

# Realizamos la predicción que van  arepresentar los datos estimdos para streams
predicciones = reg.predict(data[['in_spotify_playlists', 'in_apple_playlists']])
print(f"Ordenada al origen: {reg.intercept_}")
print(f"Pendientes: {reg.coef_}")#Es la magnitud del cambio 
plt.figure(figsize=(12, 6))# Graficmos los datos reales y las predicciones

# Decidí hacerlo por separado para que se observe mejor paara  'in_spotify_playlists'
plt.subplot(1, 2, 1)
plt.scatter(data['in_spotify_playlists'], data['streams'], label='Datos reales')
plt.plot(data['in_spotify_playlists'], predicciones, color='red', label='Regresión')
plt.title('Relación con Spotify Playlists')
plt.xlabel('in_spotify_playlists')
plt.ylabel('streams')
plt.legend()

#Para 'in_apple_playlists'
plt.subplot(1, 2, 2)
plt.scatter(data['in_apple_playlists'], data['streams'], label='Datos reales')
plt.plot(data['in_apple_playlists'], predicciones, color='red', label='Regresión')
plt.title('Relación con Apple Playlists')
plt.xlabel('in_apple_playlists')
plt.ylabel('streams')
plt.legend()

#Mostrar los gráficos
plt.tight_layout()
plt.show()
print("*******************************************************************************")
#Aquí alculamos  las predicciones
Y_test = data['streams']  # Valores reales (streams)
Y_pred = reg.predict(data[['in_spotify_playlists', 'in_apple_playlists']])  # Predicciones
print(f"Las predicciones son: {Y_pred}")
# Calcular el MSE 
mse = mean_squared_error(Y_test, Y_pred)
#Podemos ver que tiene una tendencia  positiva porque cuando aumenta in_spotify_playlists e in_apple_playlists, también lo hace streams
#Pero cosidero que como son  demasiados datos no se puede apreciar bien la relación
#Por  lo que posiblemente  no es  lineal ya que anteriormente depuramos outliers, valores nulos, etc. es decir normalizamos
# # Mostrar los resultados
print(f"MSE: {mse}")








