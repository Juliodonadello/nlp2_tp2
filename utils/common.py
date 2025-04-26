# utils/common.py

import io
import re
import warnings
from PyPDF2 import PdfReader
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import OpenAI
from langchain_core.runnables import Runnable
from typing import Any, Optional
from langchain_core.runnables.config import RunnableConfig
from thefuzz import fuzz
import os

# Model personalizado
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)

class CustomAzureChatModel(Runnable):
    def __init__(self, model="gpt-4o-mini", temperature=0.7):
        self.model = model
        self.temperature = temperature

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        messages_for_api = []
        for message in input.messages:
            if message.type == "system":
                messages_for_api.append({"role": "system", "content": message.content})
            elif message.type == "human":
                corrected_content = message.content.replace("Donadelo", "Donadello") if fuzz.ratio("Donadelo", "Donadello") > 80 else message.content
                messages_for_api.append({"role": "user", "content": corrected_content})
            elif message.type == "ai":
                messages_for_api.append({"role": "assistant", "content": message.content})

        response = client.chat.completions.create(
            messages=messages_for_api,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            top_p=1,
        )
        return response.choices[0].message.content

# Funciones auxiliares
def download_document(file_name):
    with open(file_name, 'rb') as f:
        pdf_content = f.read()

    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    text = text.replace("\n", " ").replace("\\n", " ")
    text = " ".join(text.split())
    return text

def split_text_with_langchain(text, max_length, chunk_overlap, source):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    documents = text_splitter.create_documents([text])
    for doc in documents:
        doc.metadata = {"source": source}
    indices = [f"{source}_{i+1}" for i in range(len(documents))]
    return documents, indices

def decisor_node(question):
    name_pattern = r'\b(Julio|Jose|Carlos)\b'
    matches = re.findall(name_pattern, question, re.IGNORECASE)
    selected_names = {match.capitalize() for match in matches}
    return {"plan": list(selected_names)}
