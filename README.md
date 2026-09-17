# Transfer learning para clasificar razas de gatos y perros

Proyecto chico para practicar (y poder explicar de memoria) cómo se hace
clasificación de imágenes con transfer learning en la práctica: una
**ResNet18 preentrenada en ImageNet**, con el backbone congelado y una capa
final nueva entrenada sobre 6 razas del dataset **Oxford-IIIT Pet**.

## Por qué este dataset y esta arquitectura

- **Oxford-IIIT Pet, recortado a 6 razas** (3 gatos + 3 perros), en vez del
  dataset completo (37 razas) o de un simple "gato vs. perro": con solo dos
  clases, una ResNet18 preentrenada resuelve el problema casi perfecto sin
  aprender nada interesante para mostrar. Eligiendo dos pares de razas
  visualmente parecidas (`Bengal`/`Egyptian_Mau` y
  `american_pit_bull_terrier`/`staffordshire_bull_terrier`) la matriz de
  confusión final tiene algo real que analizar.
- **ResNet18 + transfer learning** (backbone congelado, cabeza nueva
  entrenada) en vez de entrenar desde cero: es como se hace en la práctica
  cuando se tienen cientos (no millones) de imágenes por clase.

## Cómo correrlo

El notebook [`transfer_learning_mascotas.ipynb`](transfer_learning_mascotas.ipynb)
corre de punta a punta en:

- **Google Colab** (recomendado): subir el notebook, activar GPU gratuita en
  `Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU`, y
  correr todas las celdas. Tarda unos pocos minutos con GPU.
- **Localmente**, con un entorno virtual:

  ```bash
  python -m venv .venv
  .venv/Scripts/activate   # en Windows
  pip install torch torchvision matplotlib scikit-learn jupyter
  jupyter notebook transfer_learning_mascotas.ipynb
  ```

La primera vez que se corre descarga automáticamente el dataset
Oxford-IIIT Pet (~800 MB) a una carpeta `data/` local (ignorada por git).

## Qué explica el notebook

El notebook está organizado en 4 preguntas — la estructura que hay que poder
responder sobre cualquier proyecto de visión por computadora:

| Sección | Pregunta que responde |
|---|---|
| **A. Qué entra** | Cómo una imagen cruda se convierte en el tensor que ve la red (resize/crop, normalización, y por qué esos valores de mean/std específicamente) |
| **B. Qué hace el modelo** | Qué hace estructuralmente el backbone: convolución → mapas de activación → downsampling → vector de features |
| **C. Qué sale** | Cómo se pasa de logits a probabilidades por clase (softmax), mostrado sobre imágenes reales |
| **D. Qué se optimiza** | La función de pérdida (cross-entropy), el optimizador, y qué está congelado vs. entrenable durante el fine-tuning, y por qué |
| **E. Qué se analiza** | Accuracy, matriz de confusión, y ejemplos mal clasificados con una hipótesis concreta de por qué falló cada uno |

El resumen ejecutivo al principio del notebook (con los números reales de mi
propia corrida) está escrito para poder decirse en voz alta en una
entrevista.

## Resultado de mi corrida

**83.4% de accuracy** en test (586 imágenes). Los dos patrones de error
esperados aparecieron tal cual: `American Pit Bull Terrier` ↔
`Staffordshire Bull Terrier` (perros musculosos parecidos) y
`Egyptian Mau` → `Bengal` (gatos de pelaje moteado parecido), mientras que
`Pomeranian` (la raza más distinta del grupo) salió 100% correcta. Detalle
completo, con ejemplos concretos e hipótesis por error, en
[`EXPLICACION.md`](EXPLICACION.md).

## Otros archivos del proyecto

- [`EXPLICACION.md`](EXPLICACION.md) — la misma explicación conceptual del
  notebook, en texto plano, para leer sin abrir Jupyter/Colab.
- `entrenar.py` — script de validación: el mismo pipeline que el notebook,
  usado para confirmar que corre de punta a punta y para generar
  `resultados.json` con números reales antes de escribir las explicaciones.
- `construir_notebook.py` — genera `transfer_learning_mascotas.ipynb` a
  partir de `resultados.json`, para que el resumen y las hipótesis de error
  citen números reales en vez de placeholders.
