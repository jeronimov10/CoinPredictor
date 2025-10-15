from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from Contexto import contexto_simulacion_montecarlo_total, contexto_simple_texto


model = OllamaLLM(model="llama3")


template = """
Eres un experto en análisis de inversiones, dataframes y gráficas.
Tu tarea es ayudar a maximizar inversiones en BTC-USD.

Este es tu contexto:
{contexto}

Pregunta del usuario:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

retriever = csm()

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    c = retriever.invoke(question)
    result = chain.invoke({"reviews": c, "question": question})
    print(result)