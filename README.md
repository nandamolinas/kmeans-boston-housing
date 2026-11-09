# Clusterização K-Means — Boston Housing

Agrupamento de imóveis da base Boston Housing com K-Means, dividido em dois módulos.

## Estrutura

```
data/HousingData.csv        base de dados (valores ausentes como "NA")
treinar.py                  módulo de treinamento
inferir.py                  módulo de inferência
exemplo_casa.json           exemplo de nova residência
models/kmeans_boston.joblib modelo treinado (pré-processador + K-Means)
models/cotovelo.png         gráfico do método do cotovelo
requirements.txt
```

## Como executar

```bash
pip install -r requirements.txt
python treinar.py                          # treina e salva o modelo
python inferir.py --json exemplo_casa.json # infere o grupo de uma nova casa
python inferir.py                          # modo interativo
```

## Módulo de treinamento (`treinar.py`)

- **Valores ausentes** (6 colunas com 20 `NA` cada): mediana nas quantitativas e moda na categórica.
- **Normalização cuidadosa com os tipos de dado:**
  - Quantitativas (`CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT`) → `StandardScaler`.
  - Categórica (`CHAS`, 0/1) → **não é padronizada**, para preservar seu significado.
  - `MEDV` (preço) fica fora do agrupamento, pois a inferência recebe casas novas que ainda não têm preço; ela é usada apenas para descrever os grupos.
- Não há balanceamento, pois o K-Means é não supervisionado.
- **Método do cotovelo:** inércia (soma das distâncias² aos centroides) para k = 1…10; o cotovelo é detectado automaticamente como o ponto mais distante da reta que liga o primeiro ao último ponto da curva. Resultado: **k = 3**. O gráfico fica em `models/cotovelo.png`.
- O modelo final é salvo com `joblib`, junto do pré-processador ajustado.

## Módulo de inferência (`inferir.py`)

1. Recebe os dados de uma nova residência (JSON, string JSON ou entrada interativa).
2. Normaliza com o mesmo pré-processador do treino (média e desvio do treino, sem reajustar).
3. Prediz o grupo mais próximo.
4. Exibe o centroide **desnormalizado** (unidades originais) e o perfil do grupo (quantidade de imóveis e preço médio).

Campos ausentes na nova casa são imputados com a mediana/moda do treino. Em `CHAS`, o centroide mostra a proporção de imóveis do grupo que margeiam o rio.

## Resultado do treino

| Grupo | Imóveis | Preço médio (MEDV, US$ mil) | Perfil |
|-------|---------|-----------------------------|--------|
| 0 | 167 | 16,4 | alta criminalidade, TAX ≈ 623, RAD ≈ 20, LSTAT ≈ 18% |
| 1 | 254 | 24,3 | perfil intermediário, residencial |
| 2 | 85 | 29,5 | bairros mais novos (AGE ≈ 31%), muitos lotes grandes (ZN ≈ 53), criminalidade e LSTAT mais baixas, mais cômodos (RM ≈ 6,8) |
