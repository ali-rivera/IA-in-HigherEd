'''
Rag_Bot.py

Pipeline object that takes in a user query, gets relevant docs, 
passes them to the LLM and returns the response + docs to the user.

****FIND WHERE THIS CODE CAME FROM/UPDATE TO WORK****
8/16/24
'''

######################################
########### import packages ##########
######################################

import pandas as pd
from typing import Optional, List, Tuple #type hinting
import openai

from langchain_text_splitters import RecursiveCharacterTextSplitter #splitter

#langsmith setup
import os
import dotenv
dotenv.load_dotenv()

#load in Documents
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders import DirectoryLoader

from transformers import AutoTokenizer

# embedding and searching
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

#plotting
# import pacmap
import numpy as np

#for langsmith tracing
from langsmith import traceable
from langsmith.wrappers import wrap_openai

######################################
############ Create Class ############
######################################


class RagBot:

    def __init__(self, retriever, model: str = "gpt-3.5-turbo"):
        self._retriever = retriever
        # Wrapping the client instruments the LLM
        self._client = wrap_openai(openai.Client(api_key=os.getenv('OPEN_AI_KEY')))
        self._model = model

    # @traceable()
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)

    # @traceable()
    def invoke_llm(self, question, docs):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI teaching assistant with expertise in Machine Learning."
                    " Use the following docs to produce a solution to the user question."
                    "If you do not know the answer, respond with \'Couldn't tell ya\' \n\n"
                    f"## Docs\n\n{docs}",
                },
                {"role": "user", "content": question},
            ],
        )

        # Evaluators will expect "answer" and "contexts"
        return {
            "answer": response.choices[0].message.content,
            "contexts": [doc for doc in docs],
        }

    # @traceable()
    def get_answer(self, question: str):
        docs = self.retrieve_docs(question)
        return self.invoke_llm(question, docs)