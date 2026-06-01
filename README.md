# Previsão de Preço de Carros — Machine Learning P2

## Integrantes do Grupo
- **RA: 1987363** — ANDRÉ LUIS DA SILVA REIS
- **RA: 1994104** — JOSÉ VITOR DE ALMEIDA LIMA
- **RA: 1993917** — JOAQUIM FERNANDO SANTANA MOREIRA

---

## Descrição do Problema
O mercado automotivo envolve inúmeras variáveis que influenciam o preço final de um veículo. Fabricantes, revendedores e consumidores precisam de referências confiáveis para estimar valores de mercado. A precificação manual é subjetiva e suscetível a erros, o que torna a aplicação de técnicas de Machine Learning uma alternativa robusta e escalável.

## Objetivo do Projeto
Construir um modelo de regressão supervisionada capaz de prever o preço de mercado de diferentes modelos de carros com base em suas características técnicas (potência, dimensões, peso, consumo de combustível, tipo de motor, entre outras) e na marca do fabricante.

---

## Dataset Utilizado
- **Nome:** CarPrice_Assignment (1985 Automobile Dataset)
- **Arquivo:** `data/dataset.csv`
- **Registros:** 205 veículos
- **Features originais:** 26 colunas (incluindo `car_ID` e `CarName`, que foram removidas na etapa de pré-processamento)
- **Features utilizadas no modelo:** 23 variáveis preditoras (numéricas e categóricas)
- **Variável-alvo:** `price` (preço de mercado do veículo em dólares americanos)

## Tipo de Problema de Machine Learning
**Regressão supervisionada** — a variável-alvo (`price`) é contínua e numérica.

---

## Metodologia

### Análise Exploratória de Dados (EDA)
- Visualização das distribuições das variáveis numéricas e categóricas.
- Análise de correlação entre features numéricas e a variável-alvo via heatmap.
- Identificação das variáveis mais correlacionadas com `price`: `enginesize`, `curbweight`, `horsepower`, `carwidth`.

### Pré-Processamento
1. **Engenharia de Features:** extração da variável `brand` a partir do `CarName` (com padronização de nomes); criação da variável `car_age` (removida por variância zero); remoção de `car_ID`, `CarName` e `car_model`.
2. **Tratamento de Outliers:** detecção pelo método IQR na variável `price`, com remoção dos registros fora dos limites inferior e superior.
3. **Pipeline de Pré-Processamento:** utilização de `ColumnTransformer` com `StandardScaler` para variáveis numéricas e `OneHotEncoder` para variáveis categóricas, encapsulados em um `Pipeline` do scikit-learn. O fit foi aplicado exclusivamente no conjunto de treino, evitando data leakage.

### Divisão dos Dados
- **Treino:** 60%
- **Validação:** 20%
- **Teste:** 20%
- Divisão realizada com `train_test_split` e `random_state=42` para reprodutibilidade.

---

## Modelos Treinados
Foram treinados três modelos de regressão:

1. **LinearRegression** — regressão linear clássica, sem regularização.
2. **Ridge** — regressão linear com regularização L2 (`alpha=1.0`), que penaliza coeficientes elevados para reduzir overfitting por multicolinearidade.
3. **RandomForestRegressor** — modelo ensemble baseado em árvores de decisão, com otimização de hiperparâmetros via `GridSearchCV` (cv=3). Parâmetros ajustados: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.

---

## Modelo Final Escolhido
O modelo selecionado foi o **LinearRegression**, que apresentou o melhor R² no conjunto de teste (0.8866). A seleção considerou, em ordem de prioridade: R² no teste, menor RMSE e menor MAPE.

**Melhores hiperparâmetros do RandomForest (via GridSearchCV):**
- `n_estimators`: 100
- `max_depth`: 10
- `min_samples_leaf`: 2
- `min_samples_split`: 5

Apesar da otimização, o RandomForest não superou o LinearRegression neste dataset específico, indicando que a relação entre as features e o preço é predominantemente linear após o pré-processamento adequado.

---

## Métricas de Avaliação
As métricas utilizadas foram:
- **MAE** (Mean Absolute Error) — erro médio absoluto em dólares.
- **RMSE** (Root Mean Squared Error) — raiz do erro quadrático médio, penaliza erros grandes.
- **R²** (Coeficiente de Determinação) — proporção da variância explicada pelo modelo (0 a 1).
- **MAPE** (Mean Absolute Percentage Error) — erro percentual médio.

---

## Principais Resultados

### Métricas no Conjunto de Teste

| Modelo | MAE (US$) | RMSE (US$) | R² | MAPE (%) |
|:---|---:|---:|---:|---:|
| **LinearRegression** | **986,55** | **1.380,60** | **0,8866** | **10,44** |
| Ridge | 1.100,97 | 1.504,59 | 0,8653 | 11,51 |
| RandomForest | 979,98 | 1.461,59 | 0,8729 | 10,05 |

### Análise de Overfitting (R² por conjunto)

| Modelo | R² Treino | R² Validação | R² Teste |
|:---|---:|---:|---:|
| LinearRegression | 0,9646 | 0,5601 | 0,8866 |
| Ridge | 0,9488 | 0,7750 | 0,8653 |
| RandomForest | 0,9623 | 0,6555 | 0,8729 |

### K-Fold Cross-Validation (K=5)

| Modelo | Média R² | Desvio R² | Mín R² | Máx R² |
|:---|---:|---:|---:|---:|
| LinearRegression | 0,7613 | 0,1568 | 0,4541 | 0,8760 |
| **Ridge** | **0,8511** | **0,0606** | **0,7303** | **0,8884** |
| RandomForest | 0,7850 | 0,0725 | 0,6806 | 0,8620 |

> **Observação:** o Ridge apresentou o melhor desempenho no K-Fold (maior média R² e menor desvio padrão), demonstrando maior estabilidade. No entanto, o LinearRegression obteve o melhor R² absoluto no conjunto de teste, sendo selecionado como modelo final. O Pipeline salvo permite que o modelo seja facilmente substituído caso se opte pelo Ridge em cenários que priorizem estabilidade.

---

## Estrutura dos Arquivos

```text
trabalho_machine_Learning_melhorado/
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Documentação do projeto
├── .gitignore                      # Arquivos ignorados pelo Git
│
├── notebooks/
│   └── notebook_atualizado.ipynb   # Notebook revisado da P1
│
├── model/
│   └── modelo_final.pkl            # Pipeline completo (preprocessor + modelo)
│
├── reports/
│   └── relatorio_atualizado.pdf    # Relatório final atualizado
│
└── data/
    └── dataset.csv                 # Dataset utilizado
```

---

## Tecnologias Utilizadas
- **Python 3.12** — linguagem principal
- **pandas** — manipulação e análise de dados
- **NumPy** — operações numéricas
- **Matplotlib / Seaborn** — visualização de dados e gráficos
- **scikit-learn** — pré-processamento, modelagem, métricas e GridSearchCV
- **joblib** — serialização do modelo treinado
- **Streamlit** — framework para criação da aplicação web interativa
- **Jupyter Notebook** — ambiente de desenvolvimento e documentação do código

---

## Instruções para Executar o Notebook

```bash
# 1. Clone o repositório
git clone <URL_DO_REPOSITORIO>
cd trabalho_machine_Learning_melhorado

# 2. Crie um ambiente virtual e instale as dependências
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Abra o notebook
cd notebooks
jupyter notebook notebook_atualizado.ipynb
```

## Instruções para Executar o App Streamlit

```bash
# Na raiz do projeto, com o venv ativado:
streamlit run app.py
```
O navegador abrirá automaticamente na interface da aplicação.

---

## Link do App Publicado
- **Streamlit Cloud:** [https://projeto-final-machine-learning.streamlit.app](https://projeto-final-machine-learning.streamlit.app)

---

## Limitações
- O dataset contém apenas 205 registros, o que limita a capacidade de generalização dos modelos, especialmente em marcas com poucos exemplares.
- Todos os veículos são de aproximadamente 1985 — o modelo não captura depreciação temporal ou tendências de mercado atuais.
- O conjunto de validação apresentou R² significativamente inferior ao de teste em alguns modelos, sugerindo sensibilidade à divisão dos dados (dataset pequeno).
- O modelo não considera fatores como quilometragem, estado de conservação, região geográfica ou condições econômicas.

## Conclusão
O projeto atingiu seu objetivo de construir um pipeline de Machine Learning completo para prever preços de carros. Através da análise exploratória, foram identificadas as variáveis mais influentes (`enginesize`, `horsepower`, `curbweight`, `carwidth`). Após o treinamento e avaliação de três modelos, o **LinearRegression** foi selecionado como modelo final com **R² = 0,8866** e **MAE = US$ 986,55** no conjunto de teste, demonstrando boa capacidade preditiva. O modelo foi encapsulado em um Pipeline completo (pré-processamento + predição) e disponibilizado em uma aplicação Streamlit funcional, simulando um cenário real de uso.
