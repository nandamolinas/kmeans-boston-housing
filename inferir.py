"""
Módulo de INFERÊNCIA - descobre a qual grupo uma nova residência pertence.

Fluxo:
  1. Recebe os dados da nova residência (arquivo JSON, string JSON ou digitação).
  2. Normaliza com o MESMO pré-processador ajustado no treino (nunca reajusta).
  3. Prediz o grupo mais próximo (K-Means).
  4. Exibe os dados DESNORMALIZADOS do centroide desse grupo.

Uso:
    python inferir.py --json exemplo_casa.json
    python inferir.py --dados '{"CRIM":0.03,"ZN":0,"INDUS":2.3,"CHAS":0,"NOX":0.54,"RM":6.5,
                               "AGE":65,"DIS":4.1,"RAD":1,"TAX":296,"PTRATIO":15.3,"B":396.9,"LSTAT":5}'
    python inferir.py              # modo interativo (pergunta campo a campo)

Campos ausentes são tratados como valores faltantes e imputados com a
mediana (numéricas) ou a moda (CHAS) calculadas no treino.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "kmeans_boston.joblib"

DESCRICAO = {
    "CRIM": "taxa de criminalidade per capita",
    "ZN": "% de terrenos residenciais grandes (> 25 mil pés²)",
    "INDUS": "% de área comercial não varejista",
    "CHAS": "1 = margeia o rio Charles, 0 = não",
    "NOX": "concentração de óxidos nítricos",
    "RM": "nº médio de cômodos",
    "AGE": "% de imóveis anteriores a 1940",
    "DIS": "distância ponderada até centros de emprego",
    "RAD": "índice de acesso a rodovias radiais",
    "TAX": "imposto predial por US$ 10 mil",
    "PTRATIO": "razão aluno/professor",
    "B": "índice B (proporção de população negra)",
    "LSTAT": "% de população de baixa renda",
}


def carregar_modelo(caminho: Path = MODEL_PATH) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {caminho}. Execute antes: python treinar.py")
    return joblib.load(caminho)


def desnormalizar_centroide(artefato: dict, grupo: int) -> pd.Series:
    """Converte o centroide (espaço padronizado) de volta para as unidades originais."""
    pre = artefato["preprocessador"]
    centroide = artefato["kmeans"].cluster_centers_[grupo]
    cols_num = artefato["colunas_numericas"]
    cols_cat = artefato["colunas_categoricas"]

    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    valores_num = scaler.inverse_transform(centroide[: len(cols_num)].reshape(1, -1))[0]
    # Variável categórica não foi padronizada: o centroide já está na escala original
    # (média de 0/1 = proporção de imóveis do grupo com CHAS = 1).
    valores_cat = centroide[len(cols_num):]

    return pd.Series(np.concatenate([valores_num, valores_cat]), index=cols_num + cols_cat)


def inferir(dados: dict, artefato: dict | None = None) -> dict:
    """Recebe um dict com os dados de uma casa e devolve o grupo e o centroide desnormalizado."""
    artefato = artefato or carregar_modelo()
    cols = artefato["colunas_numericas"] + artefato["colunas_categoricas"]

    desconhecidas = set(dados) - set(cols)
    if desconhecidas:
        raise ValueError(f"Campos desconhecidos: {sorted(desconhecidas)}. Esperados: {cols}")
    ausentes = [c for c in cols if dados.get(c) is None]

    nova = pd.DataFrame([{c: dados.get(c, np.nan) for c in cols}], columns=cols).astype(float)

    nova_norm = artefato["preprocessador"].transform(nova)      # normaliza (usa média/desvio do treino)
    grupo = int(artefato["kmeans"].predict(nova_norm)[0])       # infere o grupo
    centroide = desnormalizar_centroide(artefato, grupo)        # volta para unidades originais

    return {
        "grupo": grupo,
        "campos_imputados": ausentes,
        "dados_normalizados": pd.Series(nova_norm[0], index=cols),
        "centroide": centroide,
        "perfil": artefato["perfil_grupos"].loc[grupo],
    }


def entrada_interativa(cols) -> dict:
    print("Informe os dados da residência (Enter = valor ausente):")
    dados = {}
    for c in cols:
        txt = input(f"  {c:8s} ({DESCRICAO.get(c, '')}): ").strip().replace(",", ".")
        dados[c] = float(txt) if txt else None
    return dados


def main():
    parser = argparse.ArgumentParser(description="Infere o grupo K-Means de uma nova residência.")
    parser.add_argument("--json", type=Path, help="arquivo JSON com os dados da residência")
    parser.add_argument("--dados", type=str, help="string JSON com os dados da residência")
    args = parser.parse_args()

    artefato = carregar_modelo()
    cols = artefato["colunas_numericas"] + artefato["colunas_categoricas"]

    if args.json:
        dados = json.loads(args.json.read_text(encoding="utf-8"))
    elif args.dados:
        dados = json.loads(args.dados)
    else:
        dados = entrada_interativa(cols)

    r = inferir(dados, artefato)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

    print("\n=== Dados da nova residência (originais) ===")
    print(pd.Series({c: dados.get(c) for c in cols}).to_string())
    if r["campos_imputados"]:
        print(f"\n[aviso] campos ausentes imputados com mediana/moda do treino: {r['campos_imputados']}")

    print("\n=== Dados normalizados ===")
    print(r["dados_normalizados"].to_string())

    print(f"\n>>> A residência pertence ao GRUPO {r['grupo']} "
          f"({int(r['perfil']['qtd_imoveis'])} imóveis no treino, "
          f"preço médio MEDV = US$ {r['perfil']['preco_medio_MEDV']:.1f} mil)")

    print(f"\n=== Centroide do grupo {r['grupo']} (desnormalizado) ===")
    print(r["centroide"].to_string())
    print("\nObs.: em CHAS, o centroide é a proporção de imóveis do grupo que margeiam o rio.")


if __name__ == "__main__":
    main()
