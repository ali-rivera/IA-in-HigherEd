'''
RAG.py

Loads in locally saved db , gets user input (Q) and 
calls Rag_Bot to retrieve content, pass Q + content to LLM, and
returns answer to the user

Author: Ali Rivera ali.rivera@virginia.edu
8/16/24
'''

######################################
################ ToDO ################
# Get user query from StreamLit UI
# fix LangSmith project (tracing)
# improve latency
######################################


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

from Rag_Bot import RagBot #local file - ensure this is in the same directory

######################################
###### load embedding model & db #####
######################################

api_key = os.getenv('OPEN_AI_KEY')

client = wrap_openai(openai.Client(api_key=api_key))

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    # model_kwargs={"device": "cuda"}, #using cpu when running locally - change if connecting to GPU for more speed
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

db = FAISS.load_local("new_faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

## create retriever for db
retriever = db.as_retriever()

######################################
######### create RAG pipeline ########
######################################

rag_bot = RagBot(retriever)

user_query = input("What is your question?\n")
response=rag_bot.get_answer(user_query)
print(response['answer'])