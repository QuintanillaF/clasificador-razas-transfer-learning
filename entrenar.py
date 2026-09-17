"""
Script de validacion local: corre el mismo pipeline que va a ir en el notebook,
para (a) confirmar que el codigo funciona de punta a punta y (b) obtener numeros
reales (accuracy, matriz de confusion, ejemplos mal clasificados) antes de
escribir las celdas de markdown que explican los resultados.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Las 6 razas elegidas: 3 gatos + 3 perros, con dos pares visualmente parecidos
# a proposito (para que la matriz de confusion tenga algo interesante que mostrar).
RAZAS_ELEGIDAS = [
    "Bengal",                        # gato atigrado/manchado
    "Egyptian Mau",                  # gato atigrado/manchado -> parecido a Bengal
    "Russian Blue",                  # gato de color solido, bien distinto a los otros dos
    "American Pit Bull Terrier",     # perro musculoso de hocico corto
    "Staffordshire Bull Terrier",    # perro musculoso de hocico corto -> parecido al anterior
    "Pomeranian",                    # perro pequeno y peludo, bien distinto al resto
]

root = Path("./data")

# Cargamos el dataset completo (37 razas) sin transformar todavia, solo para
# quedarnos con los indices que corresponden a nuestras 6 razas.
base_train = torchvision.datasets.OxfordIIITPet(
    root=str(root), split="trainval", target_types="category", download=True
)
base_test = torchvision.datasets.OxfordIIITPet(
    root=str(root), split="test", target_types="category", download=True
)

todas_las_clases = base_train.classes
indices_clases = [todas_las_clases.index(r) for r in RAZAS_ELEGIDAS]
print("Indices de clases elegidas:", list(zip(RAZAS_ELEGIDAS, indices_clases)))

# Mapeo: indice original (0-36) -> indice nuevo (0-5), para las 6 clases elegidas
mapa_indices = {orig: nuevo for nuevo, orig in enumerate(indices_clases)}


def indices_de(dataset):
    """Devuelve las posiciones (dentro del dataset) que pertenecen a nuestras 6 razas."""
    idxs = []
    for i, (_, etiqueta) in enumerate(dataset._samples if hasattr(dataset, "_samples") else []):
        pass
    return idxs


# OxfordIIITPet guarda las etiquetas accesibles via dataset._labels en algunas
# versiones; para no depender de un atributo interno, leemos parsed_images_dir
# indirectamente iterando (dataset ya cachea las imagenes en disco, es rapido).
def filtrar(dataset):
    idxs = [i for i in range(len(dataset)) if dataset._labels[i] in indices_clases]
    return idxs

idx_train_full = filtrar(base_train)
idx_test = filtrar(base_test)
print("Imagenes trainval en las 6 razas:", len(idx_train_full))
print("Imagenes test en las 6 razas:", len(idx_test))

# Separamos trainval en train/val (85/15), estratificado de forma simple por barajado fijo.
rng = random.Random(SEED)
idx_train_full_shuffled = idx_train_full[:]
rng.shuffle(idx_train_full_shuffled)
n_val = int(0.15 * len(idx_train_full_shuffled))
idx_val = idx_train_full_shuffled[:n_val]
idx_train = idx_train_full_shuffled[n_val:]
print("Train:", len(idx_train), "Val:", len(idx_val), "Test:", len(idx_test))

# --- Pipeline de entrada ---
# ResNet18 fue entrenada en ImageNet con imagenes de 224x224 normalizadas con
# la media y el desvio estandar (por canal RGB) de ese mismo dataset. Si no
# replicamos esa normalizacion, los pesos congelados del backbone reciben una
# distribucion de pixeles distinta a la que aprendieron a procesar, y las
# features que extraen dejan de ser confiables.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
tf_eval = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class SubsetConTransformYRemapeo(torch.utils.data.Dataset):
    """Envuelve el dataset original: aplica una transformacion especifica y
    remapea la etiqueta original (0-36) a la etiqueta nueva (0-5)."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, etiqueta_orig = self.dataset[self.indices[i]]
        img = self.transform(img)
        etiqueta = mapa_indices[etiqueta_orig]
        return img, etiqueta


ds_train = SubsetConTransformYRemapeo(base_train, idx_train, tf_train)
ds_val = SubsetConTransformYRemapeo(base_train, idx_val, tf_eval)
ds_test = SubsetConTransformYRemapeo(base_test, idx_test, tf_eval)

dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=0)
dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=0)
dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=0)

# --- Modelo: ResNet18 preentrenada, backbone congelado, cabeza nueva ---
pesos = ResNet18_Weights.IMAGENET1K_V1
modelo = torchvision.models.resnet18(weights=pesos)

for parametro in modelo.parameters():
    parametro.requires_grad = False

n_features = modelo.fc.in_features  # 512 para resnet18
modelo.fc = nn.Linear(n_features, len(RAZAS_ELEGIDAS))  # esta capa SI es entrenable

n_entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
n_totales = sum(p.numel() for p in modelo.parameters())
print(f"Parametros entrenables: {n_entrenables} / {n_totales} totales "
      f"({100*n_entrenables/n_totales:.3f}%)")

modelo.to(DEVICE)

criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.fc.parameters(), lr=1e-3)

# --- Entrenamiento ---
EPOCHS = 6
historial = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}


def correr_epoca(loader, entrenar: bool):
    modelo.train() if entrenar else modelo.eval()
    loss_total, correctos, n = 0.0, 0, 0
    with torch.set_grad_enabled(entrenar):
        for imgs, etiquetas in loader:
            imgs, etiquetas = imgs.to(DEVICE), etiquetas.to(DEVICE)
            if entrenar:
                optimizador.zero_grad()
            logits = modelo(imgs)
            loss = criterio(logits, etiquetas)
            if entrenar:
                loss.backward()
                optimizador.step()
            loss_total += loss.item() * imgs.size(0)
            correctos += (logits.argmax(1) == etiquetas).sum().item()
            n += imgs.size(0)
    return loss_total / n, correctos / n


for epoca in range(1, EPOCHS + 1):
    tl, ta = correr_epoca(dl_train, entrenar=True)
    vl, va = correr_epoca(dl_val, entrenar=False)
    historial["train_loss"].append(tl)
    historial["train_acc"].append(ta)
    historial["val_loss"].append(vl)
    historial["val_acc"].append(va)
    print(f"Epoca {epoca}: train_loss={tl:.3f} train_acc={ta:.3f} "
          f"val_loss={vl:.3f} val_acc={va:.3f}")

# --- Evaluacion final en test ---
test_loss, test_acc = correr_epoca(dl_test, entrenar=False)
print(f"Test: loss={test_loss:.3f} acc={test_acc:.3f}")

# Matriz de confusion + guardar predicciones para inspeccionar despues
modelo.eval()
y_true, y_pred, confianzas, rutas_idx = [], [], [], []
with torch.no_grad():
    for imgs, etiquetas in dl_test:
        imgs = imgs.to(DEVICE)
        logits = modelo(imgs)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        y_true.extend(etiquetas.tolist())
        y_pred.extend(pred.cpu().tolist())
        confianzas.extend(conf.cpu().tolist())

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print("Matriz de confusion (filas=real, columnas=predicho):")
print(RAZAS_ELEGIDAS)
print(cm)
print(classification_report(y_true, y_pred, target_names=RAZAS_ELEGIDAS))

# Guardamos todo en un JSON para poder escribir el notebook con numeros reales
resultado = {
    "razas_elegidas": RAZAS_ELEGIDAS,
    "n_train": len(idx_train),
    "n_val": len(idx_val),
    "n_test": len(idx_test),
    "n_entrenables": n_entrenables,
    "n_totales": n_totales,
    "historial": historial,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "matriz_confusion": cm.tolist(),
    "y_true": y_true,
    "y_pred": y_pred,
    "confianzas": confianzas,
}
Path("resultados.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")
print("Resultados guardados en resultados.json")

# Mal clasificados, ordenados por confianza descendente (los errores "mas seguros"
# primero, que suelen ser los mas interesantes de explicar)
mal_clasificados = [
    (i, y_true[i], y_pred[i], confianzas[i])
    for i in range(len(y_true)) if y_true[i] != y_pred[i]
]
mal_clasificados.sort(key=lambda t: -t[3])
print(f"\nTotal mal clasificados: {len(mal_clasificados)} / {len(y_true)}")
for i, real, pred, conf in mal_clasificados[:10]:
    print(f"idx_test={i} real={RAZAS_ELEGIDAS[real]} pred={RAZAS_ELEGIDAS[pred]} conf={conf:.2f}")
