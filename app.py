"""
Classificador de flores Iris com KNN - App Streamlit
Checkpoint FIAP - Machine Learning

Como rodar no computador:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =====================================================================
# CONFIGURAÇÕES GERAIS
# =====================================================================
st.set_page_config(page_title="KNN no Iris", page_icon="🌸", layout="wide")

# Nomes das colunas em português, só para ficar mais claro na tela
NOMES_COLUNAS = {
    "sepal length (cm)": "Comprimento da sépala (cm)",
    "sepal width (cm)": "Largura da sépala (cm)",
    "petal length (cm)": "Comprimento da pétala (cm)",
    "petal width (cm)": "Largura da pétala (cm)",
}
COLUNAS_X = list(NOMES_COLUNAS.values())

# O dataset guarda a espécie como número; aqui traduzimos para o nome
ESPECIES = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Uma cor para cada espécie (usada em todos os gráficos)
CORES = {"setosa": "#7B5EA7", "versicolor": "#E3A72F", "virginica": "#2E7D6B"}

# Mesmos valores de K exigidos no exercício
VALORES_K = [1, 3, 5, 7, 9]

# Atalhos para as duas colunas usadas no gráfico principal
PETALA_COMP = "Comprimento da pétala (cm)"
PETALA_LARG = "Largura da pétala (cm)"


# =====================================================================
# PARTE 1: CARREGAR E TRATAR OS DADOS (mesma lógica do Colab)
# =====================================================================
@st.cache_data  # guarda o resultado para não recarregar a cada clique
def carregar_dados():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df = df.rename(columns=NOMES_COLUNAS)
    df["especie"] = iris.target
    df["nome_especie"] = df["especie"].map(ESPECIES)

    tamanho_original = len(df)
    nulos = int(df.isnull().sum().sum())
    duplicadas = int(df.duplicated().sum())
    df = df.drop_duplicates()

    return df, tamanho_original, nulos, duplicadas


df, tamanho_original, nulos, duplicadas = carregar_dados()


# =====================================================================
# PARTE 2: PREPARAR X / y E SEPARAR TREINO / TESTE
# =====================================================================
X = df[COLUNAS_X]   # as 4 medidas (as "perguntas")
y = df["especie"]   # a espécie (o "gabarito")

# Mesmos parâmetros do notebook, para os números baterem com a apresentação.
# Se vocês colocarem stratify=y no Colab, coloquem aqui também.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


def comparar_valores_k():
    """Treina um KNN para cada K e devolve uma tabela com as acurácias."""
    linhas = []
    for valor_k in VALORES_K:
        knn = KNeighborsClassifier(n_neighbors=valor_k)
        knn.fit(X_train, y_train)

        # Validação cruzada: 10 "provas" diferentes, depois tira a média
        notas_cv = cross_val_score(
            KNeighborsClassifier(n_neighbors=valor_k), X, y, cv=10
        )

        linhas.append({
            "K": valor_k,
            "Acurácia treino": knn.score(X_train, y_train),
            "Acurácia teste": accuracy_score(y_test, knn.predict(X_test)),
            "Validação cruzada (média)": notas_cv.mean(),
        })
    return pd.DataFrame(linhas)


# =====================================================================
# BARRA LATERAL: o usuário escolhe o K e as medidas da flor nova
# =====================================================================
st.sidebar.header("Modelo")
k = st.sidebar.select_slider(
    "Número de vizinhos (K)", options=VALORES_K, value=5
)

st.sidebar.header("Medidas da flor nova")
medidas = {}
for coluna in COLUNAS_X:
    medidas[coluna] = st.sidebar.slider(
        coluna,
        min_value=float(df[coluna].min()),
        max_value=float(df[coluna].max()),
        value=round(float(df[coluna].mean()), 1),  # começa na média
        step=0.1,
    )

# Treina o modelo com o K escolhido
modelo = KNeighborsClassifier(n_neighbors=k)
modelo.fit(X_train, y_train)

# Monta a flor nova com as mesmas colunas do X e faz a previsão
flor_nova = pd.DataFrame([medidas])
previsao = modelo.predict(flor_nova)[0]
nome_previsto = ESPECIES[previsao]

# Descobre quem são os K vizinhos mais próximos e a distância até cada um
distancias, posicoes = modelo.kneighbors(flor_nova)
vizinhos = X_train.iloc[posicoes[0]].copy()
vizinhos["Espécie"] = y_train.iloc[posicoes[0]].map(ESPECIES).values
vizinhos["Distância"] = distancias[0].round(2)


# =====================================================================
# TELA PRINCIPAL
# =====================================================================
st.title("🌸 Classificador de flores Iris com KNN")
st.write(
    "Mexa nas medidas da flor na barra lateral e veja qual espécie o modelo "
    "prevê, e quais flores do treino decidiram essa previsão."
)

aba_flor, aba_dados, aba_k = st.tabs(
    ["Classificar uma flor", "Explorar os dados", "Comparar valores de K"]
)

# ---------------------------------------------------------------------
# ABA 1: classificar uma flor nova
# ---------------------------------------------------------------------
with aba_flor:
    with st.expander("Como o KNN decide?"):
        st.write(
            "Diga-me com quem andas e te direi quem és. O modelo mede a "
            "distância da flor nova até todas as flores do treino, pega as "
            f"**{k} mais próximas** e a espécie mais votada entre elas vence."
        )

    col_resultado, col_grafico = st.columns([1, 2])

    with col_resultado:
        st.metric("Espécie prevista", nome_previsto.capitalize())

        # predict_proba devolve a fração de vizinhos de cada espécie;
        # multiplicando por K, temos o número de votos
        fracoes = modelo.predict_proba(flor_nova)[0]
        votos = pd.DataFrame({
            "Espécie": [ESPECIES[c] for c in modelo.classes_],
            "Votos": (fracoes * k).round().astype(int),
        })
        st.write(f"Votação entre os {k} vizinhos:")
        st.dataframe(votos, hide_index=True)

    with col_grafico:
        fig = px.scatter(
            df,
            x=PETALA_COMP,
            y=PETALA_LARG,
            color="nome_especie",
            color_discrete_map=CORES,
            opacity=0.45,
            labels={"nome_especie": "Espécie"},
        )
        # Contorno preto nos vizinhos que votaram
        fig.add_trace(go.Scatter(
            x=vizinhos[PETALA_COMP],
            y=vizinhos[PETALA_LARG],
            mode="markers",
            marker=dict(size=15, symbol="circle-open",
                        color="black", line=dict(width=2)),
            name=f"{k} vizinhos",
        ))
        # Estrela na flor nova
        fig.add_trace(go.Scatter(
            x=[medidas[PETALA_COMP]],
            y=[medidas[PETALA_LARG]],
            mode="markers",
            marker=dict(size=20, symbol="star", color="#D7263D",
                        line=dict(width=1, color="black")),
            name="Flor nova",
        ))
        st.plotly_chart(fig)
        st.caption(
            "Os vizinhos são calculados com as 4 medidas, mas o gráfico mostra "
            "só as 2 da pétala. Por isso um vizinho pode parecer longe aqui."
        )

    st.subheader("Os vizinhos que decidiram")
    st.dataframe(vizinhos, hide_index=True)

# ---------------------------------------------------------------------
# ABA 2: exploração e tratamento dos dados (Parte 1 do Colab)
# ---------------------------------------------------------------------
with aba_dados:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas originais", tamanho_original)
    c2.metric("Valores nulos", nulos)
    c3.metric("Duplicadas removidas", duplicadas)
    c4.metric("Linhas após tratamento", len(df))

    st.subheader("Primeiras linhas")
    st.dataframe(df[COLUNAS_X + ["nome_especie"]].head(), hide_index=True)

    st.subheader("Resumo estatístico")
    st.dataframe(df[COLUNAS_X].describe().round(2))
    st.caption(
        "A espécie ficou fora do resumo de propósito: ela é uma categoria, "
        "então média e desvio padrão não fazem sentido para ela."
    )

    st.subheader("Flores por espécie")
    contagem = df["nome_especie"].value_counts().rename_axis("Espécie")
    st.dataframe(contagem.rename("Quantidade"))

    st.subheader("Compare as medidas")
    st.write(
        "Troque os eixos: na pétala as espécies se separam bem; "
        "na sépala, versicolor e virginica se misturam."
    )
    col_x, col_y = st.columns(2)
    eixo_x = col_x.selectbox("Eixo horizontal", COLUNAS_X, index=0)
    eixo_y = col_y.selectbox("Eixo vertical", COLUNAS_X, index=1)
    fig_dados = px.scatter(
        df, x=eixo_x, y=eixo_y, color="nome_especie",
        color_discrete_map=CORES, labels={"nome_especie": "Espécie"},
    )
    st.plotly_chart(fig_dados)

# ---------------------------------------------------------------------
# ABA 3: comparação dos valores de K (Parte 2 do Colab, com extras)
# ---------------------------------------------------------------------
with aba_k:
    tabela = comparar_valores_k()

    st.subheader("Tabela comparativa")
    # Mostra como porcentagem com vírgula (padrão brasileiro): 96,7%
    def porcentagem(valor):
        return f"{valor:.1%}".replace(".", ",")

    formato = {coluna: porcentagem for coluna in tabela.columns if coluna != "K"}
    st.dataframe(tabela.style.format(formato), hide_index=True)

    fig_k = px.line(
        tabela,
        x="K",
        y=["Acurácia treino", "Acurácia teste", "Validação cruzada (média)"],
        markers=True,
        labels={"value": "Acurácia", "variable": ""},
    )
    fig_k.update_layout(yaxis_tickformat=".0%")
    fig_k.update_xaxes(tickvals=VALORES_K)
    st.plotly_chart(fig_k)

    st.write(
        "Com K = 1 o treino sempre dá 100%, porque a vizinha mais próxima de "
        "cada flor do treino é ela mesma: é decoreba, não aprendizado. "
        f"O teste tem só {len(X_test)} flores, então cada erro tira cerca de "
        f"{porcentagem(1 / len(X_test))}. A validação cruzada faz 10 provas diferentes "
        "e tira a média, por isso é a medida mais confiável."
    )

    st.subheader(f"Matriz de confusão com K = {k}")
    y_pred = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    nomes = list(ESPECIES.values())
    fig_matriz = px.imshow(
        matriz,
        x=nomes,
        y=nomes,
        text_auto=True,
        color_continuous_scale="Purples",
        labels=dict(x="Espécie prevista", y="Espécie real", color="Flores"),
    )
    st.plotly_chart(fig_matriz)
    st.caption(
        "A diagonal são os acertos. Qualquer número fora dela é um erro: "
        "a linha mostra a espécie real e a coluna, o que o modelo previu."
    )
