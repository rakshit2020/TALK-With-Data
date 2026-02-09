from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
import os

loader = TextLoader(
    file_path=r"D:\DATA\TalkwithDATA\133-Annual2008_normalized.md",
    encoding="utf-8"
)

documents = loader.load()

raw_text = documents[0].page_content
import re
from langchain_core.documents import Document

def split_markdown_tables(text: str):
    """
    Splits markdown into:
    - full table blocks
    - normal text blocks
    """
    table_pattern = r"(\n\|.*?\|\n(?:\|.*?\|\n)+)"
    parts = re.split(table_pattern, text, flags=re.DOTALL)

    blocks = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            blocks.append(cleaned)
    return blocks

blocks = split_markdown_tables(raw_text)

docs = []
for block in blocks:
    docs.append(
        Document(page_content=block)
    )

print(f"Blocks created: {len(docs)}")
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language="markdown",
    chunk_size=1500,
    chunk_overlap=200
)
final_docs = []
for d in docs:
    if d.page_content.strip().startswith("|"):
        # table → keep as is
        final_docs.append(d)
    else:
        final_docs.extend(text_splitter.split_documents([d]))


print(f"Total chunks: {len(final_docs)}")

# print(final_docs)
for i in range(30):
    print(final_docs[i].page_content)
    print("\n" + "*" * 100 + "\n")

os.environ["NVIDIA_API_KEY"] = "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"

embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v2",
    truncate="END"
)


# from langchain_community.vectorstores import FAISS

print("Storing documents in FAISS vectorstore...")
vectorstore = FAISS.from_documents(
    documents=final_docs,
    embedding=embeddings
)
print(f"Successfully created vectorstore with {len(final_docs)} documents.")




