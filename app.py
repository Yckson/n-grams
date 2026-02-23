import streamlit as st
from collections import defaultdict
import re
import pandas as pd
import os
from datasets import load_dataset


x = 3

def ngrams(listOfWords, window=3):
    """
    Gera n-gramas a partir de uma lista de tokens.

    Um n-grama é uma subsequência contígua de 'n' itens de uma sequência.
    Esta função desliza uma janela de tamanho 'window' sobre a lista de tokens,
    capturando cada subconjunto consecutivo como uma tupla.

    Pseudocódigo correspondente (TRAIN-NGRAM-MODEL):
        window <- SUBSEQUENCE(tokens, i, i + n - 1)

    Exemplo:
        ngrams(["int", "main", "(", ")"], window=2)
        -> [("int", "main"), ("main", "("), ("(", ")")]

    Args:
        listOfWords: lista de tokens (strings) extraídos do texto.
        window: tamanho do n-grama, equivalente ao 'n' do pseudocódigo.

    Returns:
        Lista de tuplas, cada uma representando um n-grama.
    """
    listNgrams = []

    # Itera até o último índice onde uma janela completa cabe na lista.
    # range(0, len - window + 1) garante que não haverá janelas incompletas.
    for i in range(0, len(listOfWords) - window + 1):
        # Fatia a lista para obter a janela atual de 'window' tokens consecutivos
        windowSlice = listOfWords[i:i+window]
        # Converte a fatia em tupla (imutável e hasheável) para uso em conjuntos
        listNgrams.append(tuple(windowSlice))

    return listNgrams

def jaccard(text1, text2, window=3):
    """
    Calcula a similaridade de Jaccard entre dois textos usando n-gramas.

    O índice de Jaccard mede a sobreposição entre dois conjuntos:
        J(A, B) = |A ∩ B| / |A ∪ B|

    O processo segue os mesmos passos de pré-processamento do pseudocódigo
    (PREPROCESS-AND-TOKENIZE): os textos são normalizados para minúsculas e
    tokenizados com uma expressão regular que reconhece operadores de código-fonte
    (==, !=, <=, >=, &&, ||, ->, ::, etc.) além de palavras comuns.
    Em seguida, n-gramas são gerados sobre os tokens e a similaridade é calculada
    sobre os conjuntos de n-gramas.

    Args:
        text1: texto original (referência).
        text2: texto suspeito de plágio.
        window: tamanho do n-grama usado na comparação.

    Returns:
        Float entre 0 e 1 representando a similaridade (1 = idênticos).
    """
    # PREPROCESS: converte para minúsculas e tokeniza capturando operadores
    # compostos de código (ex: ==, !=) e símbolos de pontuação individualmente
    text1Splitted = re.findall(r"[\w]+|==|!=|<=|>=|&&|\|\||->|::|>>|<<|\+\+|--|[.,!?;(){}\[\]=+\-*/%^&|~<>:#@]", text1.lower())
    text2Splitted = re.findall(r"[\w]+|==|!=|<=|>=|&&|\|\||->|::|>>|<<|\+\+|--|[.,!?;(){}\[\]=+\-*/%^&|~<>:#@]", text2.lower())

    # Gera os conjuntos de n-gramas para cada texto.
    # O uso de set() elimina duplicatas, focando na presença de padrões,
    # não na frequência — comportamento adequado para detecção de plágio.
    set1 = set(ngrams(text1Splitted, window))
    set2 = set(ngrams(text2Splitted, window))

    # Calcula o numerador (n-gramas em comum) e o denominador (todos os n-gramas distintos)
    intersec = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # Retorna a proporção de n-gramas compartilhados sobre o total de n-gramas únicos
    return intersec / union

st.sidebar.write("Configurações")
x = st.sidebar.select_slider("N-gramas: ", options=range(1, 11))
clicked = st.sidebar.button("Executar Similaridade de plágio")
st.sidebar.divider()
st.title("Análise de semelhança bruta")
[col1, col2] = st.columns(2)


text1 = col1.text_area("Texto original")
text2 = col2.text_area("Texto suspeito")



if (text1 and text2 and clicked):
    similarity = jaccard(text1, text2, x)
    st.write(f"Similaridade: {(similarity * 100):.2f}%")


# ------------------------------------------------------------------

def codeTokenizer(code):
    """
    Tokeniza código-fonte em uma lista de tokens individuais.

    Corresponde à etapa PREPROCESS-AND-TOKENIZE do pseudocódigo. Usa uma
    expressão regular para separar palavras/identificadores de operadores e
    pontuação, produzindo a sequência de tokens que alimenta o modelo n-grama.

    Args:
        code: string contendo o código-fonte a ser tokenizado.

    Returns:
        Lista de strings, cada uma representando um token do código.
    """
    # Captura palavras (incluindo apóstrofo) e operadores/pontuação isolados
    return re.findall(r"[\w']+|[.,!?;(){}\[\]=+-/*]", code)


@st.cache_resource
def trainModel(dataset_name="bigcode/the-stack", data_dir="data/c", max_samples=500, n=3, token=None):
    """
    Treina um modelo de linguagem n-grama usando Estimativa de Máxima Verossimilhança (MLE).

    Implementa o pseudocódigo TRAIN-NGRAM-MODEL. Para cada amostra de código do
    dataset, tokeniza o conteúdo e percorre todos os n-gramas possíveis. A cada
    janela de 'n' tokens, os (n-1) primeiros formam o 'histórico' (contexto) e o
    último token é a 'próxima palavra'. As contagens são acumuladas em um dicionário
    de dicionários, que representa as estruturas 'counts' e 'context_totals' do
    pseudocódigo de forma combinada.

    Pseudocódigo correspondente:
        model <- new NGRAM-MODEL
        tokens <- PREPROCESS-AND-TOKENIZE(corpus)
        for i = 1 to LENGTH(tokens) - n + 1 do
            history <- window[1 ... n-1]
            word    <- window[n]
            INCREMENT-COUNT(model.counts, history, word)

    Nota: ADD-PADDING nao e aplicado nesta implementacao. O dataset de
    treinamento e suficientemente grande para que os primeiros tokens de cada
    arquivo nao distorcam significativamente o modelo.

    Args:
        dataset_name: identificador do dataset no HuggingFace Hub.
        data_dir: subconjunto (subset) do dataset a ser carregado.
        max_samples: numero maximo de arquivos de codigo processados.
        n: ordem do n-grama (ex: 2 = bigrama, 3 = trigrama).
        token: token de autenticacao do HuggingFace (opcional).

    Returns:
        Tupla (model, error):
            model: dicionario {historico: {proxima_palavra: contagem}}.
            error: string com mensagem de erro, ou None em caso de sucesso.
    """
    # model[history][nextWord] = contagem de ocorrencias
    # Equivale a model.counts do pseudocódigo; context_totals é derivado
    # somando os valores internos para um dado histórico.
    model = defaultdict(lambda: defaultdict(int))

    try:
        kwargs = dict(streaming=True, split="train")
        if data_dir:
            kwargs["data_dir"] = data_dir
        if token:
            kwargs["token"] = token
        # Carrega o corpus em modo streaming para nao carregar tudo em memória
        ds = load_dataset(dataset_name, **kwargs)
    except Exception as e:
        return None, str(e)

    count = 0
    for sample in ds:
        if count >= max_samples:
            break

        # Extrai o conteúdo bruto do arquivo de código (campo varia por dataset)
        code = sample.get("content", sample.get("code", ""))

        # PREPROCESS-AND-TOKENIZE: converte o código em lista de tokens
        tokens = codeTokenizer(code)

        # Percorre todos os n-gramas possíveis na sequência de tokens.
        # range(len(tokens) - n + 1) garante janelas completas de tamanho n.
        for i in range(len(tokens) - n + 1):
            # Os (n-1) tokens anteriores formam o histórico (contexto)
            history = tuple(tokens[i:i + n - 1])
            # O token na posição i + (n-1) é a próxima palavra a ser predita
            nextW = tokens[i + n - 1]
            # INCREMENT-COUNT: acumula a co-ocorrência (historico, proxima_palavra)
            model[history][nextW] += 1

        count += 1

    if not model:
        return None, "Nenhum dado encontrado no dataset."

    return model, None


def nextWord(model, currText, n=3):
    """
    Prediz os tokens mais prováveis como continuação do texto atual.

    Implementa a consulta ao modelo n-grama treinado, equivalente às funções
    GET-PROBABILITY e GENERATE-TEXT do pseudocódigo. Extrai o histórico
    (últimos n-1 tokens do texto atual), consulta o modelo e retorna os
    candidatos ordenados por contagem decrescente, o que equivale a ordenar
    por probabilidade MLE (a contagem relativa é a probabilidade).

    Pseudocódigo correspondente (GET-PROBABILITY / GENERATE-TEXT):
        current_history <- últimos (n-1) tokens de currText
        candidates      <- model.VOCABULARY()
        next_word       <- SAMPLE-FROM-DISTRIBUTION(model, candidates, current_history)

    A probabilidade de cada candidato 'w' dado o histórico 'h' é:
        P(w | h) = counts[h][w] / context_totals[h]
                 = counts[h][w] / sum(counts[h].values())

    Args:
        model: dicionário {historico: {proxima_palavra: contagem}} gerado por trainModel.
        currText: texto atual digitado pelo usuário (código parcial).
        n: ordem do n-grama (deve ser igual ao 'n' usado no treinamento).

    Returns:
        Lista de até 5 tuplas (token, contagem) ordenadas por frequência decrescente.
        Retorna lista vazia se o histórico for menor que (n-1) tokens.
    """
    # PREPROCESS-AND-TOKENIZE: converte o texto atual nos mesmos tokens do treino
    tokens = codeTokenizer(currText)
    # O histórico tem tamanho (n-1), conforme definição do n-grama
    historySize = n - 1

    # Não há histórico suficiente para consulta — retorna sem sugestões
    if (len(tokens) < historySize):
        return []

    # Extrai os últimos (n-1) tokens como contexto atual (janela deslizante)
    # Equivale a UPDATE(current_history, next_word) do pseudocódigo GENERATE-TEXT
    currHistory = tuple(tokens[-historySize:])

    # GET-PROBABILITY: busca todas as palavras seguintes observadas para este histórico
    # model.get retorna {} caso o histórico nunca tenha sido visto no treino
    predictions = model.get(currHistory, {})

    # Ordena os candidatos por contagem (maior = mais provável) — equivalente
    # a ordenar por probabilidade MLE sem necessidade de normalizar
    orderedWords = sorted(predictions.items(), key=lambda item: item[1], reverse=True)

    # Retorna as 5 sugestões mais prováveis
    return orderedWords[:5]


st.title("Predição de código")

dataset_name = st.sidebar.text_input("Dataset HuggingFace:", value="bigcode/the-stack")
data_dir = st.sidebar.text_input("Subset (data_dir):", value="data/c")
max_samples = st.sidebar.slider("Amostras para treino:", min_value=100, max_value=20000, value=500, step=100)
hf_token = st.sidebar.text_input("HuggingFace Token:", value=os.environ.get("HF_TOKEN", ""), type="password")
train_btn = st.sidebar.button("Treinar modelo")

if "model" not in st.session_state:
    st.session_state["model"] = None

if train_btn:
    with st.spinner(f"Baixando e treinando com {max_samples} amostras..."):
        model, err = trainModel(dataset_name=dataset_name, data_dir=data_dir, max_samples=max_samples, n=x, token=hf_token or None)
    if err:
        st.error(f"Erro: {err}")
    else:
        st.session_state["model"] = model
        st.success("Modelo treinado com sucesso!")

if st.session_state["model"] is not None:
    model = st.session_state["model"]

    def append_word(word):
        st.session_state["_code_widget"] = st.session_state["_code_widget"] + " " + word

    if "_code_widget" not in st.session_state:
        st.session_state["_code_widget"] = "int main"

    codingArea = st.text_area("Digite seu código C aqui:", key="_code_widget")

    if codingArea:
        suggest = nextWord(model, codingArea, n=x)
        # Debug temporário
        tokens = codeTokenizer(codingArea)
        history = tuple(tokens[-(x-1):]) if len(tokens) >= x-1 else ()
        st.write(f"Histórico buscado: `{history}` | Encontrado no modelo: `{history in model}`")

        if suggest:
            st.write("**Sugestões:**")
            cols = st.columns(len(suggest))
            for i, (word, _) in enumerate(suggest):
                cols[i].button(word, key=f"sug_{i}", on_click=append_word, args=(word,))

            words = [p for p, _ in suggest]
            count = [c for _, c in suggest]
            total = sum(count)
            prob = [round(c / total * 100, 2) for c in count]

            df = pd.DataFrame({"Probabilidade (%)": prob}, index=words)
            st.write("### Probabilidade das próximas palavras")
            st.bar_chart(df)