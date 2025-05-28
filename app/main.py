import os

from langchain_huggingface  import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader

import glob


from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

loader = DirectoryLoader("data", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")


llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    task = "text-generation"

)

import speech_recognition as sr
import pyttsx3

def listen_to_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak your question:")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            print("You said:", query)
            return query
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError:
            print("Speech recognition service is unavailable")
            return None

def speak_output(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

while True:
    user_query = listen_to_mic()

    if user_query is None:
        continue
    if user_query.lower() == "exit":
        print("Exiting. Bye!")
        break

    docs = vector_store.similarity_search(user_query, k=3)
    context = " ".join([doc.page_content for doc in docs])
    prompt = context + f"\nQuestion: {user_query}\nAnswer:"

    try:
        response = llm.invoke(prompt)
        print("\n Answer:", response)
        speak_output(response)  
    except Exception as e:
        print(" Error:", e)