"""
Apresentação do Checkpoint 2 (Dataset Iris + KNN) em Streamlit.

Como rodar (dentro da pasta do projeto):
    python -m pip install -r requirements.txt
    python -m streamlit run apresentacao.py

Navegação: botões Anterior / Próximo, setas do teclado (ou o passador de
slides) e a lista de slides na barra lateral.
"""

import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
    page_title="Dataset Iris: análise e modelo KNN",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =====================================================================
# CONFIGURAÇÕES
# =====================================================================
ESPECIES = {0: "setosa", 1: "versicolor", 2: "virginica"}
CORES = {"setosa": "#7B5EA7", "versicolor": "#E3A72F", "virginica": "#2E7D6B"}
NOMES_PT = {
    "sepal length (cm)": "Comprimento da sépala",
    "sepal width (cm)": "Largura da sépala",
    "petal length (cm)": "Comprimento da pétala",
    "petal width (cm)": "Largura da pétala",
}
COLUNAS = list(NOMES_PT.keys())
VALORES_K = [1, 3, 5, 7, 9]

# Visual igual ao da apresentação em PDF (escuro, rosa FIAP e cinza claro)
CSS = """
<style>
.stApp { background: #F6F4F5; }
[data-testid="stDecoration"], [data-testid="stAppDeployButton"] { display: none; }
/* Apresentação ocupando a tela toda: sem a faixa do Streamlit no topo e sem limite de largura */
[data-testid="stHeader"] { display: none; }
.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 100% !important; padding: 1.2rem 3rem 1rem !important; }
/* Tudo cresce junto com a tela: textos, tabelas, código e botões */
html { font-size: clamp(15px, 1vw, 20px) !important; }
/* Botões de navegação: texto numa linha só, ocupando a coluna inteira */
[data-testid="stButton"] button, .stButton button { width: 100%; white-space: nowrap; }
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: #ED145B; border-color: #ED145B; color: #fff; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
  background: #C70F4C; border-color: #C70F4C; color: #fff; }

.titulo, .card, .tabela, .nota, .bloco, .leitura, .barra, .previsao, .faixa, .contador, .texto-apoio {
  font-family: "Helvetica Neue", Arial, "Liberation Sans", sans-serif; color: #16161A;
}
.faixa { color: #6A676E; font-size: 0.85rem; margin-top: 2.5rem; }
/* Garante texto escuro nos controles, mesmo se o navegador estiver em modo escuro */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #16161A !important; font-size: 1rem; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] { color: #16161A !important; }
.legenda { color: #6A676E; font-size: 0.9rem; margin-top: 0.4rem; }
.progresso { height: 4px; background: #E3DFE1; border-radius: 2px; margin: 0.6rem 0 0.4rem; }
.progresso div { height: 100%; background: #ED145B; border-radius: 2px; }
.tabela.pequena { width: auto; font-size: 0.9rem; }
.tabela.pequena th { padding: 0.45rem 0.8rem; }
.tabela.pequena td { padding: 0.4rem 0.8rem; text-align: right; }
.tabela tbody th { background: #fff; color: #6A676E; border: 1px solid #E3DFE1; font-weight: 400; }
.grafico { background: #fff; border: 1px solid #E3DFE1; border-radius: 8px; padding: 0.5rem; }
.contador { text-align: center; color: #6A676E; font-weight: 700; }
.titulo { font-size: 2.6rem; font-weight: 700; line-height: 1.15; margin: 0.8rem 0 1.8rem; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.2rem; }
.card { background: #fff; border: 1px solid #E3DFE1; border-radius: 8px; padding: 1.5rem; }
.card .num { font-size: 3.4rem; font-weight: 700; color: #ED145B; line-height: 1; }
.card .rotulo { font-size: 1.2rem; font-weight: 700; margin-top: 1rem; }
.card .desc { color: #4A474E; margin-top: 0.4rem; font-size: 1rem; }
.card .func { font-family: "Liberation Mono", Consolas, monospace; font-weight: 700;
              color: #ED145B; font-size: 1.3rem; }
.texto-apoio { margin-top: 1.6rem; font-size: 1.1rem; }

.tabela { width: 100%; border-collapse: collapse; background: #fff; font-size: 1.05rem; }
.tabela th { background: #16161A; color: #F6F4F5; text-align: left; padding: 0.75rem 1rem; }
.tabela td { padding: 0.7rem 1rem; border: 1px solid #E3DFE1; }
.tabela td.mono { font-family: "Liberation Mono", Consolas, monospace; }

.nota { margin-bottom: 1.2rem; }
.nota .linha { color: #ED145B; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.2rem; }
.nota code { font-weight: 700; color: #16161A; background: #fff; border: 1px solid #E3DFE1;
             padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.95rem; }
.nota p { margin: 0.4rem 0 0; font-size: 1rem; line-height: 1.45; }
.rotulo-saida { font-size: 0.85rem; font-weight: 700; color: #6A676E; margin: 1rem 0 0.4rem; }
.saida { background: #fff; border: 1px solid #E3DFE1; border-radius: 8px; padding: 0.8rem 1rem;
         font-family: "Liberation Mono", Consolas, monospace; font-size: 0.9rem; white-space: pre-wrap; }
.leitura { background: #fff; border: 1px solid #E3DFE1; border-radius: 8px;
           padding: 1rem 1.2rem; margin-top: 1.2rem; }
.leitura .leitura-titulo { color: #ED145B; font-weight: 700; font-size: 1.05rem; }
.leitura p { margin: 0.4rem 0 0; line-height: 1.5; }
[data-testid="stCode"], [data-testid="stCodeBlock"] { border: 1px solid #E3DFE1; border-radius: 8px; }
[data-testid="stCode"] code, [data-testid="stCodeBlock"] code { font-size: 0.95rem; }

.bloco { border-radius: 12px; padding: 4rem; min-height: calc(100vh - 9rem);
         display: flex; flex-direction: column; justify-content: center; }
.escuro { background: #16161A; }
.escuro, .escuro b { color: #F6F4F5; }
.rosa { background: #ED145B; }
.rosa, .rosa div { color: #fff; }
.eyebrow { letter-spacing: 0.3em; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; }
.escuro .eyebrow { color: #ED145B; }
.capa-titulo { font-size: 4rem; font-weight: 700; line-height: 1.1; margin: 2rem 0 1rem; }
.capa-sub { color: #C9C5CC; font-size: 1.25rem; }
.membros { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.5rem 3rem;
           margin-top: 2.5rem; font-size: 1.05rem; }
.membros span { color: #A8A3AD; margin-left: 0.5rem; }
.frase { font-size: 2.6rem; font-weight: 700; line-height: 1.25; margin-top: 1.5rem; max-width: 22ch; }
.conclusao p { font-size: 1.3rem; margin: 0 0 1.4rem; max-width: 60ch; }
.conclusao b { color: #ED145B; }
.obrigado { font-size: 2.6rem; font-weight: 700; margin-top: 2.5rem; }

.circulos { display: flex; gap: 1.2rem; margin-bottom: 2rem; flex-wrap: wrap; }
.circulo { width: 76px; height: 76px; border-radius: 50%; background: #ED145B; color: #fff;
           font: 700 1.9rem "Helvetica Neue", Arial, sans-serif;
           display: flex; align-items: center; justify-content: center; }
.prep p { font-size: 1.15rem; margin: 0 0 1.6rem; }
.barra { border-radius: 8px; padding: 1.1rem 1.4rem; font-size: 1.15rem; margin-bottom: 1rem; }
.barra.treino { background: #ED145B; color: #fff; }
.barra.teste { background: #DAD5D9; width: 62%; min-width: 300px; }
.barra span { opacity: 0.85; margin-left: 0.4rem; }
.previsao { background: #16161A; color: #F6F4F5; border-radius: 8px; padding: 1.2rem 1.4rem; margin-top: 1rem; }
.previsao-rotulo { color: #A8A3AD; font-size: 0.9rem; }
.previsao-valor { font-size: 2.4rem; font-weight: 700; color: #ED145B; text-transform: capitalize; }
.previsao-votos { font-size: 0.95rem; margin-top: 0.3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =====================================================================
# DADOS E MODELO: exatamente a mesma lógica do notebook no Colab
# =====================================================================
@st.cache_data
def preparar():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["especie"] = iris.target
    original = df.copy()

    ausentes = int(df.isnull().sum().sum())
    duplicadas = int(df.duplicated().sum())
    pares_duplicados = df[df.duplicated(keep=False)]  # as duas cópias
    df = df.drop_duplicates()

    X = df.drop("especie", axis=1)
    y = df["especie"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    resultados = []
    for k in VALORES_K:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        resultados.append({"Valor de K (Vizinhos)": k,
                           "Acurácia": accuracy_score(y_test, y_pred)})

    return {
        "original": original, "df": df, "ausentes": ausentes,
        "duplicadas": duplicadas, "pares": pares_duplicados,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "resultados": pd.DataFrame(resultados),
    }


D = preparar()


# =====================================================================
# FUNÇÕES DE APOIO PARA MONTAR OS SLIDES
# =====================================================================
def html_bloco(conteudo):
    st.markdown(conteudo, unsafe_allow_html=True)


def titulo(texto):
    html_bloco(f'<div class="titulo">{texto}</div>')


def num(valor, casas=2):
    """Formata número com vírgula: 3.758 -> '3,76'."""
    return f"{valor:.{casas}f}".replace(".", ",")


def acuracia_txt(valor):
    """1.0 -> '1,0' e 0.9778 -> '0,978' (igual ao slide original)."""
    texto = f"{valor:.3f}".rstrip("0")
    if texto.endswith("."):
        texto += "0"
    return texto.replace(".", ",")


def cards(itens, func=False):
    """itens: lista de (destaque, rótulo, descrição)."""
    partes = ""
    for destaque, rotulo, desc in itens:
        classe = "func" if func else "num"
        rotulo_html = f'<div class="rotulo">{rotulo}</div>' if rotulo else ""
        partes += (f'<div class="card"><div class="{classe}">{destaque}</div>'
                   f'{rotulo_html}<div class="desc">{desc}</div></div>')
    html_bloco(f'<div class="cards">{partes}</div>')


def tabela(cabecalho, linhas, coluna_mono=None):
    th = "".join(f"<th>{h}</th>" for h in cabecalho)
    corpo = ""
    for linha in linhas:
        tds = "".join(
            f'<td class="{"mono" if i == coluna_mono else ""}">{c}</td>'
            for i, c in enumerate(linha)
        )
        corpo += f"<tr>{tds}</tr>"
    html_bloco(f'<table class="tabela"><thead><tr>{th}</tr></thead><tbody>{corpo}</tbody></table>')


def tabela_df(df):
    """Mostra um DataFrame como tabela HTML no visual da apresentação."""
    html_tabela = df.to_html(classes="tabela pequena", border=0, float_format=acuracia_txt)
    html_bloco(html_tabela.replace("\n", ""))


def saida(texto):
    html_bloco(f'<div class="saida">{html.escape(texto)}</div>')


def leitura(titulo_caixa, texto):
    html_bloco(f'<div class="leitura"><div class="leitura-titulo">{titulo_caixa}</div>'
               f'<p>{html.escape(texto)}</p></div>')


def slide_codigo(codigo, notas, resultado=None, caixa=None):
    """Código à esquerda (com números de linha) e explicações à direita."""
    col_codigo, col_notas = st.columns([1.3, 1], gap="large")
    with col_codigo:
        st.code(codigo, language="python", line_numbers=True)
        if resultado is not None:
            html_bloco('<div class="rotulo-saida">Resultado ao executar</div>')
            resultado()
        if caixa:
            leitura(*caixa)
    with col_notas:
        for linha, termo, texto in notas:
            html_bloco(f'<div class="nota"><div class="linha">{linha}</div>'
                       f'<code>{html.escape(termo)}</code><p>{html.escape(texto)}</p></div>')


# =====================================================================
# SLIDES (cada função desenha um slide)
# =====================================================================
def s_capa():
    membros = [
        ("Gustavo Seiji Miyakawa", "576780"), ("Guilherme Bitencourt", "575152"),
        ("Lucas Koiti Yoshikava", "574332"), ("Renata Machado", "575149"),
        ("Naiara Sousa Soares", "574792"), ("Manuela Prieto Fazzato", "576587"),
    ]
    lista = "".join(f"<div><b>{n}</b><span>RM {rm}</span></div>" for n, rm in membros)
    html_bloco(
        '<div class="bloco escuro">'
        '<div class="eyebrow">FIAP · Tecnólogo em Inteligência Artificial</div>'
        '<div class="capa-titulo">Dataset Iris:<br>análise e modelo KNN</div>'
        '<div class="capa-sub">Checkpoint 2 · Parte 1 (Estatística) e Parte 2 (Machine Learning)</div>'
        f'<div class="membros">{lista}</div></div>'
    )


def s_bibliotecas():
    titulo("Bibliotecas utilizadas")
    tabela(["Biblioteca / função", "Para que serve"], [
        ["pandas", "Organizar os dados em tabela (DataFrame)"],
        ["load_iris", "Carregar o dataset Iris original"],
        ["train_test_split", "Separar os dados em treino e teste"],
        ["KNeighborsClassifier", "Criar o modelo KNN"],
        ["accuracy_score", "Calcular a acurácia (taxa de acertos)"],
    ], coluna_mono=0)


def s_cod_importacoes():
    titulo("Código: importação das bibliotecas")
    slide_codigo(
        "# BIBLIOTECAS\n"
        "import pandas as pd\n"
        "from sklearn.datasets import load_iris\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.metrics import accuracy_score",
        [
            ("Linha 1", "# comentário",
             "Linhas que começam com # são ignoradas pelo Python. Servem para explicar o código a quem lê."),
            ("Linha 2", "import pandas as pd",
             "Importa o pandas inteiro com o apelido pd. Por isso, no resto do código, escrevemos pd.DataFrame."),
            ("Linhas 3 a 6", "from ... import ...",
             "Traz do scikit-learn só a função que vamos usar. Cada uma vem de um módulo: datasets "
             "(dados prontos), model_selection (divisão dos dados), neighbors (o KNN) e metrics (avaliação)."),
        ],
    )


def s_coleta():
    titulo("Coleta e estrutura dos dados")
    linhas, colunas = D["original"].shape
    cards([
        (linhas, "linhas", "Cada linha é uma flor"),
        (colunas, "colunas", "4 medidas + a espécie"),
        (3, "espécies", "0 Setosa · 1 Versicolor · 2 Virginica"),
    ])
    html_bloco('<div class="texto-apoio">Carregamos com <b>load_iris()</b>, transformamos em DataFrame do pandas '
               'e criamos a coluna <b>especie</b>.</div>')


def s_cod_coleta():
    titulo("Código: coleta dos dados")
    linhas, colunas = D["original"].shape
    slide_codigo(
        "# 1. Coleta dos dados\n"
        "iris = load_iris()\n"
        "\n"
        "df = pd.DataFrame(iris.data, columns=iris.feature_names)\n"
        "df['especie'] = iris.target\n"
        "\n"
        "# 2. Exploração da estrutura e características dos dados\n"
        "print(f\"-> Tamanho original do dataset: {df.shape[0]} \"\n"
        "      f\"linhas e {df.shape[1]} colunas.\")",
        [
            ("Linha 2", "load_iris()",
             "Carrega o dataset com três partes: data (as medidas), feature_names (nomes das colunas) "
             "e target (a espécie)."),
            ("Linha 4", "pd.DataFrame(...)",
             "Monta a tabela com as medidas, usando os nomes como cabeçalho das colunas."),
            ("Linha 5", "df['especie'] = iris.target",
             "Cria a coluna especie. Ela vem em números (0, 1 e 2) porque o modelo só trabalha com números."),
            ("Linhas 8 e 9", "df.shape",
             "Devolve (linhas, colunas): [0] são as linhas e [1] as colunas. "
             "O f antes das aspas encaixa esses valores no texto."),
        ],
        resultado=lambda: saida(f"-> Tamanho original do dataset: {linhas} linhas e {colunas} colunas."),
    )


def s_resumo():
    titulo("Resumo estatístico (describe)")
    resumo = D["original"][COLUNAS].describe()
    linhas = []
    for col in COLUNAS:
        r = resumo[col]
        linhas.append([NOMES_PT[col], num(r["mean"]), num(r["std"]), num(r["50%"]),
                       num(r["min"], 1), num(r["max"], 1)])
    tabela(["Medida (cm)", "Média", "Desvio padrão", "Mediana", "Mínimo", "Máximo"], linhas)


def s_cod_exploracao():
    titulo("Código: exploração dos dados")
    r = D["original"]["petal length (cm)"].describe()
    slide_codigo(
        "print(\"\\n-> Primeiras 5 linhas do dataset:\")\n"
        "display(df.head())\n"
        "\n"
        "print(\"\\n-> Resumo estatístico (Média, Mínimo, Máximo):\")\n"
        "display(df.describe())",
        [
            ("Linha 2", "display(df.head())",
             "head() pega as 5 primeiras linhas, para conferir se a tabela foi montada certo. display() "
             "mostra o resultado como tabela formatada no Colab; com print, sairia como texto simples."),
            ("Linha 5", "df.describe()",
             "Calcula, para cada coluna: quantidade (count), média (mean), desvio padrão (std, o quanto os "
             "valores se espalham), mínimo, quartis 25%, 50% e 75% e máximo. O 50% é a mediana."),
        ],
        resultado=lambda: tabela_df(D["original"].head()),
        caixa=("O que os números mostram",
               f"O comprimento da pétala tem o maior desvio padrão ({num(r['std'])}) e a média "
               f"({num(r['mean'])}) fica bem abaixo da mediana ({num(r['50%'])}). Isso indica um grupo de "
               "flores com pétalas bem menores puxando a média para baixo: as setosas. Ou seja, a pétala "
               "deve ser a medida que mais separa as espécies."),
    )


def s_ausentes():
    titulo("Valores ausentes e duplicidades")
    cards([
        (D["ausentes"], "Valores ausentes", "Verificado com df.isnull().sum()"),
        (D["duplicadas"], "Linha duplicada", "Encontrada com df.duplicated().sum()"),
        ("✓", "Tratamento", "Removida com df.drop_duplicates()"),
    ])


def s_cod_ausentes():
    titulo("Código: ausentes e duplicidades")

    def mostrar():
        saida(f"-> Valores ausentes (nulos) encontrados: {D['ausentes']}\n"
              f"-> Linhas duplicadas encontradas: {D['duplicadas']}\n"
              "-> AÇÃO: Linhas duplicadas foram removidas com sucesso!")
        html_bloco('<div class="rotulo-saida">A flor repetida (as duas cópias)</div>')
        tabela_df(D["pares"])

    slide_codigo(
        "# 3. Identificação e tratamento de valores ausentes e duplicidades\n"
        "valores_ausentes = df.isnull().sum().sum()\n"
        "linhas_duplicadas = df.duplicated().sum()\n"
        "\n"
        "# Tratamento: Se houver linhas duplicadas, o código irá removê-las\n"
        "if linhas_duplicadas > 0:\n"
        "    df = df.drop_duplicates()\n"
        "    print(\"-> AÇÃO: Linhas duplicadas foram removidas com sucesso!\")",
        [
            ("Linha 2", "df.isnull().sum().sum()",
             "isnull() marca onde falta valor. O 1º sum() conta por coluna e o 2º soma tudo."),
            ("Linha 3", "df.duplicated().sum()",
             "Marca as linhas idênticas a uma anterior (a primeira cópia não conta)."),
            ("Linha 6", "if linhas_duplicadas > 0:",
             "O tratamento só roda se existir duplicada. O recuo de 4 espaços indica o que está dentro do if."),
            ("Linha 7", "df = df.drop_duplicates()",
             f"Cria a tabela sem a cópia e guarda de novo em df. O dataset passa a ter {len(D['df'])} flores."),
        ],
        resultado=mostrar,
    )


def s_conclusao_parte1():
    html_bloco(
        '<div class="bloco rosa"><div class="eyebrow">Conclusão da Parte 1</div>'
        '<div class="frase">Os dados foram explorados e validados: o dataset está limpo e '
        'pronto para Machine Learning.</div></div>'
    )


def s_preparacao():
    titulo("Preparação: treino e teste")
    n_treino, n_teste = len(D["X_train"]), len(D["X_test"])
    col_texto, col_barras = st.columns([1, 1.1], gap="large")
    with col_texto:
        html_bloco(
            '<div class="prep nota">'
            '<p><b>X</b> = todas as colunas, exceto especie</p>'
            '<p><b>y</b> = a coluna especie (o gabarito)</p>'
            '<p><b>random_state = 42</b> garante que o sorteio seja sempre o mesmo</p></div>'
        )
    with col_barras:
        html_bloco(
            f'<div class="barra treino"><b>Treino · 70%</b><span>{n_treino} flores, para aprender</span></div>'
            f'<div class="barra teste"><b>Teste · 30%</b>'
            f'<span>{n_teste} flores, para avaliar</span></div>'
        )


def s_cod_separacao():
    titulo("Código: separação de treino e teste")
    n_treino, n_teste = len(D["X_train"]), len(D["X_test"])
    slide_codigo(
        "# 1. Preparação dos dados\n"
        "X = df.drop('especie', axis=1)\n"
        "y = df['especie']\n"
        "\n"
        "# 2. Separação dos conjuntos de treino e teste\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.3, random_state=42)",
        [
            ("Linha 2", "df.drop('especie', axis=1)",
             "Copia a tabela sem a coluna especie: ficam só as 4 medidas. axis=1 indica coluna "
             "(axis=0 seria linha)."),
            ("Linha 3", "df['especie']",
             "Pega só a coluna da espécie: o gabarito que o modelo precisa acertar."),
            ("Linha 6", "train_test_split(...)",
             "Sorteia as flores e devolve 4 partes: medidas e gabarito de treino, medidas e gabarito de teste."),
            ("Linha 7", "test_size=0.3, random_state=42",
             f"30% vão para teste: das {len(D['df'])} flores, {n_treino} treinam e {n_teste} testam. "
             "O 42 fixa o sorteio, então o resultado se repete."),
        ],
        resultado=lambda: saida(f"X_train: {n_treino} flores  |  X_test: {n_teste} flores"),
    )


def s_treinamento():
    titulo("Treinamento com diferentes valores de k")
    circulos = "".join(f'<div class="circulo">{k}</div>' for k in VALORES_K)
    html_bloco(f'<div class="circulos">{circulos}</div>')
    cards([
        ("fit", "", "Treina com os dados de treino"),
        ("predict", "", "Faz a predição nos dados de teste"),
        ("accuracy_score", "", "Calcula a taxa de acertos"),
    ], func=True)


def s_cod_loop():
    titulo("Código: o loop de treinamento")
    slide_codigo(
        "# 3. Treinamento do modelo testando diferentes valores de k\n"
        "valores_k = [1, 3, 5, 7, 9]\n"
        "resultados = []\n"
        "\n"
        "for k in valores_k:\n"
        "    knn = KNeighborsClassifier(n_neighbors=k)\n"
        "    knn.fit(X_train, y_train)\n"
        "    y_pred = knn.predict(X_test)\n"
        "    acuracia = accuracy_score(y_test, y_pred)\n"
        "    resultados.append({'Valor de K (Vizinhos)': k,\n"
        "                       'Acurácia': acuracia})",
        [
            ("Linha 5", "for k in valores_k:",
             "Repete o bloco recuado uma vez para cada K da lista."),
            ("Linha 6", "KNeighborsClassifier(n_neighbors=k)",
             "Cria o modelo dizendo quantos vizinhos vão votar."),
            ("Linha 7", "knn.fit(...)",
             "No KNN, treinar é só guardar as flores de treino. Todo o trabalho acontece na previsão."),
            ("Linha 8", "knn.predict(X_test)",
             "Para cada flor do teste, acha as K vizinhas mais próximas no treino. A mais votada é a resposta."),
            ("Linhas 9 a 11", "accuracy_score(...)",
             "Compara as respostas com o gabarito (acertos ÷ total). O append guarda o resultado na lista."),
        ],
    )


def s_tabela():
    titulo("Tabela comparativa de resultados")
    res = D["resultados"]
    col_tab, _ = st.columns([1, 1])
    with col_tab:
        tabela(["Valor de k (vizinhos)", "Acurácia"],
               [[k, acuracia_txt(a)] for k, a in zip(res["Valor de K (Vizinhos)"], res["Acurácia"])])


def s_cod_tabela():
    titulo("Código: tabela de resultados")
    n_teste = len(D["X_test"])
    slide_codigo(
        "# 4. Apresentação dos resultados em tabela comparativa\n"
        "tabela_resultados = pd.DataFrame(resultados)\n"
        "display(tabela_resultados)",
        [
            ("Linha 2", "pd.DataFrame(resultados)",
             "Transforma a lista em tabela: cada dicionário vira uma linha e cada chave vira uma coluna."),
            ("Linha 3", "display(...)", "Mostra a tabela formatada no Colab."),
        ],
        resultado=lambda: tabela_df(D["resultados"]),
        caixa=("Por que 1,0 em todos os K?",
               "O Iris é um dataset fácil: a setosa se separa sozinha das outras espécies. E o teste tem só "
               f"{n_teste} flores, então cada erro tiraria cerca de {num(100 / n_teste, 1)} pontos. Com todos "
               "os K em 100%, essa divisão não permite apontar o melhor K. Para escolher com segurança, o "
               "próximo passo seria a validação cruzada, que repete a prova com várias divisões diferentes."),
    )


def s_demo():
    titulo("Demonstração: classificando uma flor nova")
    df, X_train, y_train = D["df"], D["X_train"], D["y_train"]
    col_controles, col_grafico = st.columns([1, 1.7], gap="large")

    with col_controles:
        k = st.select_slider("Número de vizinhos (K)", options=VALORES_K, value=5, key="demo_k")
        medidas = {}
        for col in COLUNAS:
            medidas[col] = st.slider(
                f"{NOMES_PT[col]} (cm)",
                float(df[col].min()), float(df[col].max()),
                round(float(df[col].mean()), 1), 0.1,
                key=f"demo_{col}",
            )

        modelo = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        flor = pd.DataFrame([medidas])  # mesmas colunas, na mesma ordem do X
        prevista = ESPECIES[modelo.predict(flor)[0]]
        votos = (modelo.predict_proba(flor)[0] * k).round().astype(int)
        placar = ", ".join(f"{ESPECIES[c]} {v}" for c, v in zip(modelo.classes_, votos))
        html_bloco(
            f'<div class="previsao"><div class="previsao-rotulo">Espécie prevista</div>'
            f'<div class="previsao-valor">{prevista}</div>'
            f'<div class="previsao-votos">Votos dos {k} vizinhos: {placar}</div></div>'
        )

    with col_grafico:
        _, posicoes = modelo.kneighbors(flor)
        vizinhos = X_train.iloc[posicoes[0]]
        pontos = df.assign(Espécie=df["especie"].map(ESPECIES))
        fig = px.scatter(
            pontos, x="petal length (cm)", y="petal width (cm)", color="Espécie",
            color_discrete_map=CORES, opacity=0.75, template="plotly_white",
            labels={"petal length (cm)": "Comprimento da pétala (cm)",
                    "petal width (cm)": "Largura da pétala (cm)"},
        )
        fig.update_traces(marker=dict(size=10))  # pontos das flores maiores
        fig.add_trace(go.Scatter(
            x=vizinhos["petal length (cm)"], y=vizinhos["petal width (cm)"], mode="markers",
            marker=dict(size=15, symbol="circle-open", color="#16161A", line=dict(width=2)),
            name=f"{k} vizinhos",
        ))
        fig.add_trace(go.Scatter(
            x=[medidas["petal length (cm)"]], y=[medidas["petal width (cm)"]], mode="markers",
            marker=dict(size=22, symbol="star", color="#ED145B", line=dict(width=1, color="#16161A")),
            name="Flor nova",
        ))
        fig.update_layout(
            height=580, margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            font=dict(family="Helvetica Neue, Arial, sans-serif", size=17, color="#16161A"),
            legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=dict(text="")),
        )
        fig.update_xaxes(gridcolor="#ECE8EB", zeroline=False)
        fig.update_yaxes(gridcolor="#ECE8EB", zeroline=False)
        # theme=None: usa as cores definidas aqui, e não o tema (claro/escuro) do Streamlit
        st.plotly_chart(fig, theme=None)
        html_bloco('<div class="legenda">Os vizinhos são calculados com as 4 medidas, mas o gráfico '
                   'mostra só as 2 da pétala. Por isso um vizinho pode parecer longe aqui.</div>')


def s_conclusao():
    res = D["resultados"]
    acuracias = res["Acurácia"]
    if acuracias.nunique() == 1:
        parte2 = f"o KNN alcançou acurácia de {acuracia_txt(acuracias.iloc[0])} em todos os valores de k testados."
    else:
        melhor = res.loc[acuracias.idxmax()]
        parte2 = (f"a melhor acurácia foi {acuracia_txt(melhor['Acurácia'])}, "
                  f"com k = {int(melhor['Valor de K (Vizinhos)'])}.")
    html_bloco(
        '<div class="bloco escuro conclusao">'
        '<div class="titulo" style="color:#F6F4F5">Conclusão</div>'
        '<p><b>Parte 1:</b> dados sem valores ausentes, duplicidade tratada, dataset pronto para uso.</p>'
        f'<p><b>Parte 2:</b> {parte2}</p>'
        '<div class="obrigado">Obrigado! Perguntas?</div></div>'
    )


SLIDES = [
    ("Capa", s_capa),
    ("Bibliotecas utilizadas", s_bibliotecas),
    ("Código: importação das bibliotecas", s_cod_importacoes),
    ("Coleta e estrutura dos dados", s_coleta),
    ("Código: coleta dos dados", s_cod_coleta),
    ("Resumo estatístico", s_resumo),
    ("Código: exploração dos dados", s_cod_exploracao),
    ("Valores ausentes e duplicidades", s_ausentes),
    ("Código: ausentes e duplicidades", s_cod_ausentes),
    ("Conclusão da Parte 1", s_conclusao_parte1),
    ("Preparação: treino e teste", s_preparacao),
    ("Código: separação de treino e teste", s_cod_separacao),
    ("Treinamento com diferentes valores de k", s_treinamento),
    ("Código: o loop de treinamento", s_cod_loop),
    ("Tabela comparativa de resultados", s_tabela),
    ("Código: tabela de resultados", s_cod_tabela),
    ("Demonstração ao vivo", s_demo),
    ("Conclusão", s_conclusao),
]
N = len(SLIDES)


# =====================================================================
# NAVEGAÇÃO
# =====================================================================
if "slide" not in st.session_state:
    st.session_state.slide = 0


def mudar(passo):
    st.session_state.slide = min(max(st.session_state.slide + passo, 0), N - 1)


col_lista, _, col_ant, col_prox = st.columns([3.2, 3, 1.2, 1.2], vertical_alignment="center")
with col_lista:
    # Lista para pular direto para um slide (útil na hora das perguntas)
    st.selectbox(
        "Ir para o slide",
        list(range(N)),
        format_func=lambda i: f"{i + 1} de {N}: {SLIDES[i][0]}",
        key="slide",
        label_visibility="collapsed",
    )
col_ant.button("Anterior", on_click=mudar, args=(-1,), disabled=st.session_state.slide == 0)
col_prox.button("Próximo", on_click=mudar, args=(1,), type="primary",
                disabled=st.session_state.slide == N - 1)

# Barra fina mostrando quanto da apresentação já passou
progresso = (st.session_state.slide + 1) / N
html_bloco(f'<div class="progresso"><div style="width:{progresso:.0%}"></div></div>')

# Desenha o slide atual
SLIDES[st.session_state.slide][1]()

# Rodapé, igual ao da apresentação em PDF
html_bloco('<div class="faixa">Checkpoint 2 · Machine Learning &amp; Modelling · '
           'Statistical Computing</div>')

# Setas do teclado / passador de slides: clicam nos botões Anterior e Próximo.
# (Ignora as setas quando o foco está num slider, para não trocar de slide sem querer.)
components.html(
    """
    <script>
    const doc = window.parent.document;
    if (!doc.getElementById("navegacao-teclado")) {
      const s = doc.createElement("script");
      s.id = "navegacao-teclado";
      s.textContent = `
        document.addEventListener("keydown", function (e) {
          const el = document.activeElement;
          if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" ||
                     el.getAttribute("role") === "slider")) return;
          let alvo = null;
          if (e.key === "ArrowRight" || e.key === "PageDown") alvo = "Próximo";
          if (e.key === "ArrowLeft" || e.key === "PageUp") alvo = "Anterior";
          if (!alvo) return;
          const botao = Array.from(document.querySelectorAll("button"))
            .find(b => b.innerText.trim() === alvo);
          if (botao && !botao.disabled) { e.preventDefault(); botao.click(); }
        });
      `;
      doc.head.appendChild(s);
    }
    </script>
    """,
    height=0,
)
