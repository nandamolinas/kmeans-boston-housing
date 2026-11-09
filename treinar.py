"""
Módulo de TREINAMENTO - Clusterização K-Means na base Boston Housing.

Etapas:
  1. Carrega o CSV (valores ausentes vêm como 'NA').
  2. Pré-processa separando variáveis QUANTITATIVAS e CATEGÓRICAS:
       - quantitativas -> imputação pela mediana + padronização (StandardScaler)
       - categórica (CHAS, 0/1) -> imputação pela moda, SEM padronização
         (padronizar uma variável categórica destruiria o seu significado).
  3. Método do cotovelo: calcula a inércia (soma das distâncias quadráticas aos
     centroides) para k = 1..K_MAX, detecta o "cotovelo" e salva o gráfico.
  4. Treina o K-Means com o k ótimo e salva o modelo em models/kmeans_boston.joblib.

Uso:
    python treinar.py
    python treinar.py --k 4        # força um valor de k (ignora a detecção automática)
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # permite gerar o gráfico sem interface gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "HousingData.csv"
MODEL_PATH = BASE_DIR / "models" / "kmeans_boston.joblib"
ELBOW_PATH = BASE_DIR / "models" / "cotovelo.png"

RANDOM_STATE = 42
K_MAX = 10

# CHAS = 1 se o imóvel faz divisa com o rio Charles (variável categórica binária).
COLUNAS_CATEGORICAS = ["CHAS"]
# MEDV (preço) é a variável-alvo do problema supervisionado original: não entra
# no agrupamento, para que a inferência funcione com os dados de uma casa nova
# (que normalmente ainda não tem preço). Ela é usada só para descrever os grupos.
COLUNA_PRECO = "MEDV"
COLUNAS_NUMERICAS = [
    "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]


def construir_preprocessador() -> ColumnTransformer:
    """Quantitativas: mediana + StandardScaler. Categórica: moda, sem escala."""
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    # A ordem dos transformers define a ordem das colunas de saída: numéricas, depois categóricas.
    return ColumnTransformer([
        ("num", num, COLUNAS_NUMERICAS),
        ("cat", cat, COLUNAS_CATEGORICAS),
    ])


def encontrar_cotovelo(ks, inercias) -> int:
    """Detecta o cotovelo: ponto da curva mais distante da reta que liga o
    primeiro ao último ponto (curva normalizada para [0, 1])."""
    x = np.array(ks, dtype=float)
    y = np.array(inercias, dtype=float)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return int(ks[int(np.argmax(dist))])


def main():
    parser = argparse.ArgumentParser(description="Treina o K-Means na base Boston Housing.")
    parser.add_argument("--k", type=int, default=None, help="número de grupos (padrão: método do cotovelo)")
    args = parser.parse_args()

    # 1. Carregamento ---------------------------------------------------------
    df = pd.read_csv(CSV_PATH, na_values=["NA"])
    print(f"Base carregada: {df.shape[0]} registros, {df.shape[1]} colunas")
    print("Valores ausentes por coluna:\n", df.isna().sum()[df.isna().sum() > 0].to_string(), "\n")

    X = df[COLUNAS_NUMERICAS + COLUNAS_CATEGORICAS]

    # 2. Pré-processamento ----------------------------------------------------
    pre = construir_preprocessador()
    X_proc = pre.fit_transform(X)

    # 3. Método do cotovelo ---------------------------------------------------
    ks = list(range(1, K_MAX + 1))
    inercias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_proc)
        inercias.append(km.inertia_)
        print(f"k={k:2d}  inércia={km.inertia_:10.2f}")

    k_otimo = args.k if args.k else encontrar_cotovelo(ks, inercias)
    origem = "informado manualmente" if args.k else "método do cotovelo"
    print(f"\nNúmero de grupos escolhido: {k_otimo} ({origem})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, inercias, "o-", color="steelblue")
    plt.axvline(k_otimo, color="red", linestyle="--", label=f"k ótimo = {k_otimo}")
    plt.xlabel("Número de grupos (k)")
    plt.ylabel("Inércia (soma das distâncias² aos centroides)")
    plt.title("Método do cotovelo - K-Means (Boston Housing)")
    plt.xticks(ks)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()
    print(f"Gráfico do cotovelo salvo em {ELBOW_PATH.relative_to(BASE_DIR)}")

    # 4. Treino final e salvamento -------------------------------------------
    kmeans = KMeans(n_clusters=k_otimo, n_init=10, random_state=RANDOM_STATE).fit(X_proc)

    rotulos = kmeans.labels_
    perfil = (
        df.assign(grupo=rotulos)
        .groupby("grupo")
        .agg(qtd_imoveis=(COLUNA_PRECO, "size"), preco_medio_MEDV=(COLUNA_PRECO, "mean"))
        .round(2)
    )
    print("\nTamanho e preço médio (MEDV, em US$ 1000) de cada grupo:\n", perfil.to_string())

    artefato = {
        "preprocessador": pre,
        "kmeans": kmeans,
        "colunas_numericas": COLUNAS_NUMERICAS,
        "colunas_categoricas": COLUNAS_CATEGORICAS,
        "k": k_otimo,
        "perfil_grupos": perfil,
    }
    joblib.dump(artefato, MODEL_PATH)
    print(f"\nModelo salvo em {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
