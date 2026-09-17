"""
Re-analisis de los resultados del modelo usando pandas, en vez de listas
sueltas. Es el mismo analisis que ya esta en el notebook (matriz de
confusion + errores mas seguros), pero escrito con las herramientas que
se usan todos los dias para manejar datasets y metadatos (groupby, merge,
value_counts) -- justo lo que un rol de Data Scientist Trainee que cura y
analiza datasets de imagenes (con sus metadatos) necesita mostrar.

No hace falta reentrenar nada: lee resultados.json, que ya tiene guardado
y_true, y_pred y las confianzas de la corrida real.
"""
import json

import pandas as pd

res = json.loads(open("resultados.json", encoding="utf-8").read())
razas = res["razas_elegidas"]

df = pd.DataFrame({
    "real": [razas[i] for i in res["y_true"]],
    "prediccion": [razas[i] for i in res["y_pred"]],
    "confianza": res["confianzas"],
})
df["correcto"] = df["real"] == df["prediccion"]

print("--- Accuracy por raza (groupby + mean de una columna booleana) ---")
print(df.groupby("real")["correcto"].mean().round(3).sort_values())

print("\n--- Pares de confusion mas frecuentes (solo filas con error) ---")
errores = df[~df["correcto"]]
print(
    errores.groupby(["real", "prediccion"]).size()
    .sort_values(ascending=False)
    .head(5)
)

print("\n--- Confianza promedio: aciertos vs errores ---")
print(df.groupby("correcto")["confianza"].mean().round(3))
# Interpretacion: si la confianza promedio de los ERRORES es mas baja que
# la de los ACIERTOS, el modelo esta "medianamente calibrado" -- cuando no
# esta seguro, es mas probable que se este equivocando. No es perfecto
# (hay errores con confianza alta, ver seccion E.1 del notebook), pero la
# tendencia general es la esperable.

print("\n--- Resumen general con describe() ---")
print(df["confianza"].describe().round(3))
