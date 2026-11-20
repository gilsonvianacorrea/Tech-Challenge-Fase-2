# Tech Challenge - Fase 2 (FIAP)
Pipeline MLOps para previsão de intenção de compra em e-commerce usando Python, DVC e Poetry.

# Tech Challenge - Fase 2 (FIAP)
Pipeline MLOps reprodutível para previsão de intenção de compra em e-commerce.

## 1. Visão Geral do Projeto
Este projeto implementa um pipeline MLOps completo utilizando **Python**, **Poetry** para gerenciamento de dependências e **DVC (Data Version Control)** para orquestração e reprodutibilidade de dados, treinamento, avaliação e geração de relatórios visuais.

## 2. Arquitetura e Tecnologias
- **Linguagem:** Python 3.12+
- **Gerenciamento de Ambiente:** Poetry
- **Orquestração de Pipeline:** DVC (`dvc.yaml`)
- **Machine Learning & Métricas:** Scikit-Learn, Pandas, NumPy
- **Rastreamento:** Git + DVC

## 3. Estrutura do Repositório
```text
.
├── data/
│   ├── raw/                 # Dados brutos controlados pelo DVC
│   └── processed/           # Dados limpos e divididos (treino/teste)
├── models/                  # Artefatos do modelo treinado (.joblib)
├── metrics/                 # Métricas de desempenho em JSON
├── reports/
│   └── figures/             # Gráficos (Matriz de Confusão e Curva ROC)
├── src/
│   ├── data.py              # Pré-processamento e split
│   ├── train.py             # Treinamento do modelo (Random Forest)
│   ├── evaluate.py          # Cálculo de métricas
│   └── plots.py             # Geração de gráficos visuais
├── dvc.yaml                 # Definição das etapas do pipeline
├── params.yaml              # Parâmetros centrais do projeto
└── pyproject.toml           # Dependências do Poetry

Pra rodar do 0 em máquina local:

EXECUTAR NO PROMPT DO 'D.O.S' OS SEGUINTES COMANDOS

1 - Clonar o repositório e instalar dependências:
poetry install --no-root

2 - Reproduzir o pipeline completo via DVC:
poetry run dvc repro

Resultados e Avaliação do Modelo
O modelo foi avaliado no conjunto de teste, obtendo os seguintes indicadores principais (salvos em metrics/scores.json):

Acurácia (Accuracy): ~89.78%

Precisão (Precision): ~73.21%

Recall: ~53.66%

F1-Score: ~61.93%

ROC-AUC: ~91.79%

Visualizações de Desempenho
Os gráficos gerados automaticamente pelo pipeline ilustram o comportamento do modelo:

Matriz de Confusão: reports/figures/confusion_matrix.png

Curva ROC: reports/figures/roc_curve.png