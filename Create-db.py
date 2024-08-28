'''
Create-db.py

Pulls in files from /processed and chunks/embeds them, 
then stores them in a (local) vecotrized db.

Author: Ali Rivera ali.rivera@virginia.edu
8/16/24
'''

####################################
########## import packages #########
####################################

from tqdm.notebook import tqdm #progress bar
import pandas as pd
from typing import Optional, List, Tuple #type hinting
# from datasets import Dataset #to load in premade example datasets
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", None)  # This will be helpful when visualizing retriever outputs

from langchain_text_splitters import RecursiveCharacterTextSplitter #splitter

#langsmith setup
import os
import dotenv
dotenv.load_dotenv()

#load in Documents
from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter #alt import
from langchain_community.document_loaders import DirectoryLoader

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# embedding and searching
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

####################################

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

## load in docs (must be in IA-in-HigherEd dir)
loader = DirectoryLoader('./RAG-docs/processed/', glob="**/*.txt", show_progress = True) #all .txt files in processed folder
docs = loader.load()
docs

####################################
############ chunk docs ############
####################################

# save as LangChain docs (formatting)
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc.page_content, metadata= doc.metadata) for doc in tqdm(docs)]

# set embedding model name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# split (chunk) docs with chunk size = max seq length()
seq_len = SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length
print(f"Model's maximum sequence length: {seq_len}")


def split_docs(chunk_size: int,
                    knowledge_base: List[LangchainDocument],
                    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


docs_processed = split_docs(
    seq_len,  # We choose a chunk size adapted to our model
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)

####################################
##### vectorize (embed) chunks #####
####################################
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    # model_kwargs={"device": "cuda"}, #using cpu when running locally - change if connecting to GPU for more speed
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

#edit distance strategy for use case
KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

#save db locally
KNOWLEDGE_VECTOR_DATABASE.save_local("new_faiss_index")