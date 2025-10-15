from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """

Responde la pregunta de abajo.

Esta es la historia o contexto de la conversacion: {context}

Pregunta: {question}

Respuesta: 
"""

model = OllamaLLM(model="llama3")

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def conversacion():
    context = ""

    print("Bienvenido... ¿Cual es tu pregunta? escribe 'exit' para salir")
    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context":context, "question": user_input})
        print("Bot", result)

        context += f"\nUser: {user_input}\nAI: {result}"

if __name__=="__main__":
    conversacion()



