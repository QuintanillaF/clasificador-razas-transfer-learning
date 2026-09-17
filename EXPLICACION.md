# Cómo funciona este proyecto (para leer antes de tocar código)

Este documento es la explicación conceptual completa del proyecto —
pensada para poder leerse de punta a punta y, después, explicarla de
memoria en una entrevista. El notebook (`transfer_learning_mascotas.ipynb`)
tiene el mismo contenido pero ejecutable, con gráficos e imágenes reales.

> **Actualización:** el entrenamiento ya corrió. Accuracy final en test:
> **83.4%** (586 imágenes nunca vistas por el modelo). El error más
> frecuente, por lejos, fue confundir **American Pit Bull Terrier** con
> **Staffordshire Bull Terrier** (39 de 100 Pit Bulls terminaron
> clasificados como Staffordshire) — el segundo patrón más claro fue
> **Egyptian Mau → Bengal** (25 de 97 Egyptian Mau). Ambos son exactamente
> los dos pares de razas elegidos a propósito por ser visualmente
> parecidos, así que el modelo se equivocó justo donde se esperaba que se
> equivocara. `Pomeranian` salió con accuracy perfecta (100/100): es la
> raza más distinta visualmente de las 6, así que no hay con qué
> confundirla. Ver la sección final de este documento para el detalle.

---

## La idea en una frase

Tomamos una red neuronal (**ResNet18**) que ya sabe "ver" — fue entrenada
con 1.2 millones de fotos de ImageNet para reconocer 1000 categorías de
objetos — le **congelamos** todo lo que ya sabe, y le enseñamos **solo una
capa nueva** para que, en vez de 1000 categorías genéricas, reconozca 6
razas puntuales de gatos y perros. Eso es **transfer learning**: transferir
conocimiento ya aprendido en un problema grande a un problema chico y
específico, en vez de aprender todo desde cero.

## El dataset: Oxford-IIIT Pet, recortado a 6 razas

Dataset completo: 37 razas de gatos y perros, ~200 fotos por raza. Usamos
solo 6, elegidas a propósito:

| Raza | Tipo | Por qué se eligió |
|---|---|---|
| `Bengal` | gato | pelaje moteado/atigrado |
| `Egyptian Mau` | gato | pelaje moteado/atigrado — **parecido a Bengal** |
| `Russian Blue` | gato | color sólido liso — bien distinto a los otros dos gatos |
| `American Pit Bull Terrier` | perro | musculoso, hocico corto |
| `Staffordshire Bull Terrier` | perro | musculoso, hocico corto — **parecido al anterior** |
| `Pomeranian` | perro | chico y muy peludo — bien distinto al resto |

**Por qué no "gato vs. perro" simplemente:** con solo 2 clases muy distintas
entre sí, una ResNet18 preentrenada resuelve el problema casi perfecto y no
queda nada interesante para analizar en la matriz de confusión. Con dos
pares de razas visualmente parecidas a propósito, el modelo va a cometer
errores *reales* y explicables, que es justo lo que hace falta mostrar.

---

## A. QUÉ ENTRA — de imagen cruda a tensor normalizado

Cada foto pasa por 4 pasos antes de llegar a la red:

1. **Resize / Crop a 224×224 píxeles.** ResNet18 fue entrenada con ese
   tamaño exacto; si le diéramos otro tamaño, las dimensiones internas de
   la red no coincidirían con lo que espera en cada capa.
   - En **entrenamiento**: `RandomResizedCrop` (recorta una región de
     tamaño y posición aleatoria) + `RandomHorizontalFlip` (espejado
     horizontal al azar). Esto es **aumento de datos (data augmentation)**:
     con pocas fotos por clase, mostrarle a la red pequeñas variaciones de
     cada imagen en cada época reduce el sobreajuste (que memorice fotos
     puntuales en vez de aprender el concepto general de la raza).
   - En **validación/test**: `Resize` + `CenterCrop`, sin aleatoriedad —
     para medir el modelo de forma reproducible, no con suerte.
2. **ToTensor()**: convierte la imagen (píxeles 0-255) a un tensor de
   PyTorch con valores en [0, 1].
3. **Normalize(mean, std)**: le resta la media y divide por el desvío
   estándar, canal por canal (R, G, B).

### ¿Por qué mean=`[0.485, 0.456, 0.406]` y std=`[0.229, 0.224, 0.225]`, y no 0.5/0.5?

Porque son la media y el desvío estándar de los píxeles de **ImageNet**, el
dataset con el que se preentrenó ResNet18. Los filtros convolucionales del
backbone —que vamos a dejar **congelados**, sin reentrenar— aprendieron a
reconocer patrones asumiendo que los píxeles de entrada llegan centrados en
esa distribución específica. Si normalizáramos con otros valores, cada capa
del backbone recibiría activaciones corridas de rango respecto a lo que
aprendió a esperar, y las features que extrae dejan de ser confiables. No es
un detalle estético: es hacer que la entrada calce con lo que el backbone
realmente aprendió a procesar.

---

## B. QUÉ HACE EL MODELO — la arquitectura, por dentro

ResNet18 es una pila de **bloques convolucionales residuales**. La lógica
estructural, capa a capa:

1. **Convolución**: un filtro chico (p. ej. 3×3) se desliza sobre la imagen
   calculando, en cada posición, una combinación lineal de los píxeles que
   tiene debajo. El resultado es un **mapa de activación (feature map)**:
   una imagen nueva donde cada píxel indica "qué tanto se parece este
   parche a lo que este filtro busca" (un borde, una textura, un color
   puntual). Cada capa tiene muchos filtros en paralelo: de 3 canales (RGB)
   se pasa a decenas o cientos de mapas de activación.
2. **Downsampling (stride/pooling)**: cada tanto, el mapa de activación se
   reduce a la mitad de alto y ancho. Esto hace que las capas siguientes
   "vean" una región cada vez más grande de la imagen original con el mismo
   tamaño de filtro (aumenta el **campo receptivo**), y evita que la
   cantidad de cómputo explote.
3. **Más profundidad = features más abstractas**: las primeras capas
   detectan bordes y manchas de color; las del medio combinan eso en
   texturas y formas (orejas, ojos, patrones de pelaje); las capas finales
   representan conceptos de alto nivel (¿hay una cabeza de animal con estas
   proporciones y esta textura?).
4. **Global Average Pooling**: al final de los bloques convolucionales
   quedan mapas de activación de 7×7 con 512 canales. En vez de aplanarlos
   (lo que daría un vector enorme y atado a la resolución de entrada), se
   promedia cada mapa de 7×7 a **un solo número por canal** → un
   **vector de 512 números**. Ese vector es el resumen final de la imagen:
   "qué tan presente está cada uno de los 512 patrones aprendidos, en esta
   imagen en particular". Es la representación que el modelo usa para
   decidir la clase.
5. **La cabeza (`fc`)**: una capa lineal que toma ese vector de 512 números
   y lo convierte en 6 números (uno por raza) — los **logits** (ver
   sección C).

### Por qué preentrenada, y no entrenar desde cero

Entrenar una red convolucional desde cero necesita **millones** de imágenes
etiquetadas para que los filtros de las primeras capas aprendan solos a
detectar bordes y texturas básicas. Con unos cientos de fotos por clase eso
es imposible: el modelo memorizaría el dataset sin generalizar. La solución
estándar en la industria es transfer learning: reusar filtros generales ya
aprendidos en un dataset enorme (ImageNet) y entrenar solo la parte final,
específica al problema nuevo.

---

## C. QUÉ SALE — de logits a probabilidades

La capa `fc` devuelve **logits**: 6 números reales sin restricción (pueden
ser negativos, no suman 1, todavía no son probabilidades). Para convertirlos
en algo interpretable se aplica **softmax**:

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Esto hace dos cosas a la vez: vuelve todos los valores positivos (por el
exponencial) y los normaliza para que sumen exactamente 1 — el resultado es
una distribución de probabilidad real sobre las 6 clases. La predicción
final es la clase con mayor probabilidad (`argmax`).

En el notebook esto se muestra de forma concreta: para 3-4 imágenes reales
de test, se grafica la foto junto con un gráfico de barras de las 6
probabilidades que le asignó el modelo — no solo un número de accuracy
abstracto, sino "así es como el modelo ve esta foto puntual".

---

## D. QUÉ SE OPTIMIZA

### Congelado vs. entrenable

`requires_grad = False` le dice a PyTorch "no calcules gradiente para este
parámetro, no lo actualices". Se lo ponemos a **todo el backbone**: sus
pesos quedan fijos exactamente como llegaron de ImageNet. Después
reemplazamos `fc` por una capa lineal nueva (inicializada al azar), que sí
tiene `requires_grad = True` — es la única parte del modelo que aprende.
En este proyecto esa capa nueva representa bien por debajo del 1% de los
parámetros totales del modelo.

**Por qué esta división es el estándar con poca data:** el backbone ya sabe
extraer features útiles y generales; lo único que no sabe es cómo
*combinar* esas features para nuestras 6 razas específicas, y eso es lo que
tiene que aprender la cabeza. Entrenar también el backbone (fine-tuning
completo) con pocos cientos de imágenes por clase tiene alto riesgo de
sobreajustar o de "olvidar" (catastrophic forgetting) las features generales
que lo hacen útil en primer lugar. Congelarlo también hace el entrenamiento
mucho más rápido: solo hay que calcular gradientes para una fracción mínima
de los parámetros.

### La función de pérdida: Cross-Entropy Loss

Para clasificación multiclase, cross-entropy compara la distribución de
probabilidad que predice el modelo (después de softmax) contra la clase
correcta, y castiga con una penalidad **cada vez mayor cuanto más confiada y
equivocada** está la predicción. Matemáticamente es
`-log(probabilidad asignada a la clase correcta)`: si esa probabilidad es
cercana a 1, la pérdida es casi 0; si es cercana a 0, la pérdida crece sin
límite. Es la elección estándar para clasificación de una sola clase por
ejemplo porque minimizarla es exactamente equivalente a maximizar la
probabilidad que el modelo le asigna a la respuesta correcta (máxima
verosimilitud).

### El optimizador: Adam

Se aplica **solo** a los parámetros de la cabeza nueva (`modelo.fc`). En
cada paso, Adam mira el gradiente de la pérdida respecto a esos pesos —qué
tanto y en qué dirección cambiaría la pérdida si se moviera ligeramente
cada peso— y los actualiza en la dirección que la reduce, con un tamaño de
paso que ajusta automáticamente por parámetro.

### El loop de entrenamiento, paso a paso, en cada batch

1. Las imágenes pasan por el modelo → `logits` (**forward pass**).
2. Se calcula `loss = CrossEntropyLoss(logits, etiquetas_reales)`.
3. `loss.backward()`: PyTorch calcula automáticamente el gradiente de la
   pérdida respecto a cada parámetro entrenable (**backpropagation**).
4. `optimizador.step()`: cada parámetro entrenable se actualiza restándole
   (una fracción de) su gradiente.
5. `optimizador.zero_grad()`: se resetean los gradientes antes del próximo
   batch (si no, se acumularían de un batch al siguiente).

Después de cada época se mide también la pérdida y el accuracy en un set de
**validación** (que el modelo no usa para actualizar pesos), para detectar
sobreajuste: si la pérdida de entrenamiento sigue bajando pero la de
validación empieza a subir, el modelo está memorizando en vez de
generalizar.

---

## E. QUÉ SE ANALIZA

Un solo número de accuracy esconde en qué se equivoca el modelo. Por eso se
evalúa en 3 niveles:

1. **Accuracy global** en el set de test — imágenes que el modelo nunca
   vio, ni para entrenar ni para elegir hiperparámetros.
2. **Matriz de confusión**: cuenta, para cada raza real, en qué raza la
   clasificó el modelo. La diagonal son los aciertos; todo lo que está
   fuera de la diagonal es un error específico (raza real → raza
   predicha), y el patrón de esos errores es mucho más informativo que el
   accuracy solo.
3. **Ejemplos individuales mal clasificados**, con una hipótesis concreta
   de por qué el modelo se equivocó en cada caso. Esto es lo que en la
   práctica se llama *error analysis*, y es el hábito que distingue a
   alguien que solo corre modelos de alguien que entiende qué están
   haciendo — literalmente lo que este tipo de puesto busca evaluar.

Para que las hipótesis de error no sean genéricas ("se confundió, ponele"),
el notebook anota primero los rasgos visuales característicos de cada raza
(tipo de pelaje, tamaño, forma del cuerpo) y arma la hipótesis de cada error
comparando esos rasgos: si la raza real y la predicha comparten rasgos
(mismo tipo de pelaje moteado, mismo tipo de cuerpo musculoso), la confusión
probablemente es real y visual; si no se parecen en nada, el error
probablemente se debe a la imagen puntual (mala pose, recorte, iluminación)
y no a similitud entre razas.

---

## Resultados reales de la corrida

**Configuración:** 6 épocas, semilla fija (42), backbone congelado,
505 imágenes de entrenamiento / 88 de validación / 586 de test, CPU (sin GPU).

**Curva de entrenamiento** (train_acc → val_acc por época): pasó de 46%/78%
en la época 1 a 92%/86-89% en la época 6 — subida rápida porque solo se
entrena una capa lineal chica (3078 parámetros, el **0.028%** de los
11.2 millones de parámetros totales del modelo).

**Accuracy final en test: 83.4%** (más precisamente, 83.1%-83.4% según la
corrida — hay una variación mínima normal de punto flotante entre ejecuciones).

**Reporte por clase (precision / recall):**

| Raza | Precision | Recall |
|---|---|---|
| Pomeranian | 1.00 | 1.00 |
| Russian Blue | 0.97 | 0.94 |
| Egyptian Mau | 0.95 | 0.71 |
| Bengal | 0.79 | 0.98 |
| American Pit Bull Terrier | 0.72 | 0.61 |
| Staffordshire Bull Terrier | 0.63 | 0.75 |

**Matriz de confusión — lectura de los dos patrones de error:**

1. **American Pit Bull Terrier → Staffordshire Bull Terrier** (39 de 100
   Pit Bulls mal clasificados así) y, en menor medida, al revés (22 de 89
   Staffordshire clasificados como Pit Bull). Es el par de razas "bully"
   elegido a propósito por su parecido — cuerpo musculoso, pelaje corto,
   cabeza ancha — y es, con diferencia, el error más frecuente de todo el
   modelo.
2. **Egyptian Mau → Bengal** (25 de 97 Egyptian Mau mal clasificados así).
   El otro par elegido a propósito — ambos gatos de pelaje moteado/atigrado.
3. **Pomeranian: 100/100 correctas.** La raza visualmente más distinta del
   grupo (pelaje largo y esponjoso, tamaño chico) no se confunde con nada.

Esto confirma la hipótesis de diseño del proyecto: el modelo no está
memorizando razas puntuales, está usando textura de pelaje y forma general
del cuerpo (justo lo que un backbone de ImageNet sabe extraer), y por eso
falla exactamente donde dos razas comparten esos rasgos.

**5 ejemplos concretos mal clasificados** (los que el modelo predijo con
más confianza y aun así se equivocó — los "errores más seguros de sí
mismos", que son los más interesantes de explicar):

| # | Real | Predicho | Confianza | Hipótesis |
|---|---|---|---|---|
| 1 | American Pit Bull Terrier | Staffordshire Bull Terrier | 92% | Comparten cuerpo musculoso, pelaje corto liso y cabeza ancha — la diferencia real entre estas razas es sutil (proporciones, altura) y fácil de perder en un crop de 224×224. |
| 2 | American Pit Bull Terrier | Staffordshire Bull Terrier | 90% | Mismo patrón que el ejemplo 1. |
| 3 | Egyptian Mau | Bengal | 90% | Ambos tienen pelaje moteado sobre fondo claro; sin referencia de tamaño corporal en la foto, el patrón de manchas domina la decisión del modelo. |
| 4 | Egyptian Mau | Bengal | 83% | Mismo patrón que el ejemplo 3. |
| 5 | Egyptian Mau | Bengal | 81% | Mismo patrón que el ejemplo 3. |

El notebook genera esta misma tabla de forma dinámica (no hardcodeada) cada
vez que se corre, comparando los rasgos visuales anotados de cada raza —
así que si corrés el notebook de nuevo y salen imágenes distintas como
"peores errores", la hipótesis que se imprime sigue siendo válida porque
está atada al par de razas, no a la imagen puntual.
