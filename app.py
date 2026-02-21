import streamlit as st
from collections import defaultdict
import re
import pandas as pd
import os
from datasets import load_dataset


x = 3

def ngrams(listOfWords, window=3):
    listNgrams = []

    for i in range(0, len(listOfWords) - window + 1):
        windowSlice = listOfWords[i:i+window]
        listNgrams.append(tuple(windowSlice))

    return listNgrams

def jaccard (text1, text2, window=3):

    text1Splitted = re.findall(r"[\w]+|==|!=|<=|>=|&&|\|\||->|::|>>|<<|\+\+|--|[.,!?;(){}\[\]=+\-*/%^&|~<>:#@]", text1.lower())
    text2Splitted = re.findall(r"[\w]+|==|!=|<=|>=|&&|\|\||->|::|>>|<<|\+\+|--|[.,!?;(){}\[\]=+\-*/%^&|~<>:#@]", text2.lower())

    set1 = set(ngrams(text1Splitted, window))
    set2 = set(ngrams(text2Splitted, window))

    intersec = len(set1.intersection(set2))
    union = len(set1.union(set2))

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

def codeTokenizer (code):
    return re.findall(r"[\w']+|[.,!?;(){}\[\]=+-/*]", code)


@st.cache_resource
def trainModel(dataset_name="bigcode/the-stack", data_dir="data/c", max_samples=500, n=3, token=None):
    model = defaultdict(lambda: defaultdict(int))

    try:
        kwargs = dict(streaming=True, split="train")
        if data_dir:
            kwargs["data_dir"] = data_dir
        if token:
            kwargs["token"] = token
        ds = load_dataset(dataset_name, **kwargs)
    except Exception as e:
        return None, str(e)

    count = 0
    for sample in ds:
        if count >= max_samples:
            break
        code = sample.get("content", sample.get("code", ""))
        tokens = codeTokenizer(code)
        for i in range(len(tokens) - n + 1):
            history = tuple(tokens[i:i + n - 1])
            nextW = tokens[i + n - 1]
            model[history][nextW] += 1
        count += 1

    if not model:
        return None, "Nenhum dado encontrado no dataset."

    return model, None


def nextWord (model, currText, n=3):
    tokens = codeTokenizer(currText)
    historySize = n - 1

    if (len(tokens) < historySize):
        return []

    currHistory = tuple(tokens[-historySize:])
    predictions = model.get(currHistory, {})

    orderedWords = sorted(predictions.items(), key=lambda item: item[1], reverse=True)


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