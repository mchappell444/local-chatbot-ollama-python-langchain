"""
A Locally hosted chatbot, based on any Ollama hosted LLM, Python and LangChain

Uses a virtual env called "chatbot"  
"""


from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# Basic template used to prompt the LLM
TEMPLATE = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# using llama3 here a a good match for availabkle syst resources, but can use any model
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(TEMPLATE)
chain = prompt | model


def handle_conversation():
    """
    handle the conversation
    """

    context = ""
    print("Welcome to Mark's AI ChatBot. Type 'exit' to quit.")
    while True:
        user_input = input("What you want?: ")
        if user_input.lower() == "exit":
            break

        # Prompt the LLM by utilising the template, that incorporates the question plus previous session context
        result = chain.invoke(
            {"context": context, "question": user_input})

        print("Bot: ", result)

        # retain context for this conversation
        context += f"\nUser: {user_input}\nAI: {result}"


if __name__ == "__main__":
    handle_conversation()
