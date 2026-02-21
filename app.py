import streamlit as st
import os
import glob
from collections import defaultdict
import re
import pandas as pd


x = 3
filesPath = "/dataset"

def ngrams(listOfWords, window=3):
    listNgrams = []

    for i in range (0, len(listOfWords)):
        windowSlice = listOfWords[i:window-1]
        listNgrams.append(tuple(windowSlice))
    
    return listNgrams

def jaccard (text1, text2, window=3):

    text1Splitted = text1.lower().split()
    text2Splitted = text2.lower().split()

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
    st.write(similarity)


# ------------------------------------------------------------------

def codeTokenizer (code):
    return re.findall(r"[\w']+|[.,!?;(){}\[\]=+-/*]", code)


@st.cache_resource
def trainModel (sourcePath="/dataset", n=3):
    
    model = defaultdict (lambda: defaultdict(int))

    search = os.path.join(sourcePath, "*.c")
    cFiles = glob.glob(search)

    if not cFiles:
        return None

    for file in cFiles:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
            tokens = codeTokenizer(code)

            for i in range (0, len(tokens) - n + 1):
                history = tuple(tokens[i:i+n-1])
                nextW = tokens[i+n-1]
                model[history][nextW] += 1
    
    return model


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
filesPath = st.sidebar.text_input("Caminho do dataset: ")

if "_code_widget" not in st.session_state:
    st.session_state["_code_widget"] = "int main"

if filesPath:
    with st.spinner("Lendo arquivos locais e treinando modelo..."):
        N_GRAMAS = x
        model = trainModel(filesPath, n=N_GRAMAS)

    if model is None:
        st.error(f"Nenhum arquivo .c encontrado na pasta: {filesPath}")
    else:
        st.success("Modelo treinado com sucesso!")

        def append_word(word):
            st.session_state["_code_widget"] = st.session_state["_code_widget"] + " " + word

        codingArea = st.text_area("Digite seu código C aqui:", key="_code_widget")

        if codingArea:
            suggest = nextWord(model, codingArea, n=N_GRAMAS)

            if suggest:
                st.write("Sugestões:")
                cols = st.columns(len(suggest))
                for i, (word, _) in enumerate(suggest):
                    cols[i].button(word, key=f"sug_{i}", on_click=append_word, args=(word,))

                words = [p for p, _ in suggest]
                count = [c for _, c in suggest]
                total = sum(count)
                prob = [round(c / total * 100, 2) for c in count]

                df = pd.DataFrame({"Probabilidade (%)": prob}, index=words)
                st.write("### Probabilidade das próximas words")
                st.bar_chart(df)