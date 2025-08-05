# Challenge-2-Telecom
Continuacion de Ejercicio de Telecom
README TELECOM 2 ALURA LATAM

Predicción de Abandono de Clientes (Churn) en Telecomunicaciones
📝 Descripción del Proyecto
Este proyecto se centra en construir y evaluar modelos de Machine Learning para predecir el abandono de clientes (Churn) en una empresa de telecomunicaciones. El objetivo principal es identificar a los clientes con alta probabilidad de irse ( evasión), comprender los factores que impulsan su decisión y proponer estrategias de retención basadas en estos hallazgos.
🎯 Objetivo
El objetivo principal es:
•	Desarrollar un modelo predictivo que pueda identificar a los clientes en riesgo de Churn con alta precisión y recall.
•	Analizar las variables más influyentes en el Churn para obtener insights accionables.
•	Proponer estrategias de negocio basadas en estos insights para mejorar la retención de clientes.
📊 Dataset
El análisis se realiza sobre el archivo datos_limpios.csv, el cual contiene información de clientes de una compañía de telecomunicaciones, incluyendo datos demográficos, servicios contratados, información de facturación y, crucialmente, si el cliente ha abandonado la empresa (Churn_numeric).
🛠️ Metodología y Pasos Realizados
El proyecto sigue un flujo de trabajo estándar de Machine Learning:
1.	Carga y Limpieza de Datos Inicial:
o	Carga del dataset datos_limpios.csv.
o	Manejo de valores perdidos (NaN) en la columna account.Charges.Total.
o	Eliminación de la columna customerID (identificador único no relevante para el modelado).
o	Filtrado de registros con Churn_numeric igual a -1.
2.	Preprocesamiento de Datos:
o	Codificación One-Hot: Conversión de variables categóricas (como tipo de contrato, servicio de internet, método de pago, etc.) a un formato numérico utilizando pd.get_dummies (resultando en 31 características finales).
3.	División de Datos:
o	El dataset se dividió en conjuntos de entrenamiento (70%) y prueba (30%) utilizando train_test_split, asegurando la estratificación por la variable Churn_numeric para mantener la proporción de clases.
4.	Manejo de Desbalance de Clases (SMOTE):
o	Debido al desbalance inherente en los datos de Churn (menos clientes que abandonan que los que se quedan), se aplicó SMOTE (Synthetic Minority Over-sampling Technique) al conjunto de entrenamiento. Esto creó muestras sintéticas de la clase minoritaria (Churn) para equilibrar el dataset y mejorar el aprendizaje del modelo.
5.	Entrenamiento y Evaluación de Modelos:
o	Se entrenaron y evaluaron tres modelos de clasificación populares:
	Regresión Logística: Un modelo lineal que predice la probabilidad de Churn.
	Random Forest Classifier: Un modelo de conjunto robusto que combina múltiples árboles de decisión.
	Árbol de Decisión: Un modelo interpretable que crea reglas de decisión.
o	Cada modelo fue evaluado utilizando métricas clave para problemas de desbalance de clases: Matriz de Confusión, Reporte de Clasificación (Precision, Recall, F1-Score para la clase 'Churn') y AUC-ROC Score. Todos los modelos se entrenaron con el conjunto de datos balanceado por SMOTE y se evaluaron en el conjunto de prueba original.
6.	Selección del Modelo:
o	Se comparó el rendimiento de los tres modelos utilizando las métricas mencionadas, prestando especial atención a la precisión, recall, F1-Score para la clase 'Churn' y el AUC-ROC.
o	El Modelo de Regresión Logística fue seleccionado como el mejor en esta configuración (31 características), ofreciendo el mejor equilibrio de métricas para la predicción de Churn.
7.	Análisis de Importancia de Variables (Regresión Logística):
o	Se extrajeron y analizaron los coeficientes del modelo de Regresión Logística para identificar las características con mayor impacto en la probabilidad de Churn, y si su impacto era positivo (aumenta Churn) o negativo (disminuye Churn).
8.	Propuesta de Estrategias de Retención:
o	Se formularon estrategias de negocio accionables basadas en los insights obtenidos del análisis de las variables más relevantes.
📈 Resultados Clave y Modelo Seleccionado
Después de una evaluación comparativa con 31 características consistentes, el modelo de Regresión Logística demostró ser el más efectivo.
Métricas del Modelo de Regresión Logística (para la clase 'Churn'):
•	Precision: 0.56
•	Recall: 0.65
•	F1-Score: 0.60
•	AUC-ROC: 0.8228
•	Accuracy Global: 0.77
Variables Más Relevantes Identificadas:
Las variables que más influyen en la probabilidad de Churn (en orden de impacto) son:
•	account.Contract_Month-to-month: Principal impulsor de Churn.
•	internet.InternetService_Fiber optic: Los usuarios de fibra óptica tienen mayor Churn.
•	account.PaymentMethod_Electronic check: Este método de pago se asocia con mayor riesgo.
•	customer.tenure: A mayor antigüedad, menor probabilidad de Churn (factor protector).

o	💡 Estrategias de Retención Propuestas
Basadas en el análisis del modelo de Regresión Logística, se proponen las siguientes estrategias:
1.	Incentivar Contratos a Largo Plazo: Diseñar ofertas atractivas para que los clientes mensuales migren a planes anuales o bienales.
2.	Mejorar la Experiencia de Fibra Óptica: Investigar problemas de calidad o soporte para usuarios de fibra óptica.
3.	Optimizar Métodos de Pago: Fomentar el uso de métodos de pago automáticos (tarjeta, transferencia) sobre cheques electrónicos.
4.	Programas de Bienvenida y Fidelización Temprana: Ofrecer atención y beneficios especiales a clientes nuevos.

🚀 Cómo se  Ejecuto el Proyecto:
Para replicar este análisis en Google Colab:
1.	Abre el “archivo” .ipynb en Google Colab.
2.	Se sube el archivo datos_limpios.csv al entorno de Colab (hazer clic en el icono de carpeta en el panel izquierdo, luego en el icono de "subir archivo").
3.	Ejecutar la celda de instalación de librerías al inicio:
Python
!pip install imbalanced-learn
(Es crucial ejecutar esta celda primero, especialmente si el entorno de Colab se reinicia, ya que las librerías instaladas no persisten).
4.	Ejecutar las celdas de código secuencialmente de arriba a abajo.
💻 Tecnologías Utilizadas
•	Python 3
•	Pandas: Para manipulación y análisis de datos.
•	Numpy: Para operaciones numéricas.
•	Scikit-learn: Para construcción y evaluación de modelos de Machine Learning (Regresión Logística, Random Forest, Árbol de Decisión).
•	Imbalanced-learn (imblearn): Para manejar el desbalance de clases con SMOTE.
•	Matplotlib: Para visualización de datos.
•	Seaborn: Para visualización de datos.

