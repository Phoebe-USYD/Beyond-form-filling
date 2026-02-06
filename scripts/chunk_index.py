import json
import os
import tiktoken
from llama_index.core.schema import TextNode
from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.llm = None 
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en")
Settings.embed_model = embed_model 

def count_tokens(text, model="gpt-3.5-turbo"):
    """Calculate no. of tokens by tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Read through my cleaned data
with open("who_ready_for_chunking.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# Turn each section into document
docs = []
for item in json_data:
    title = item["title"]
    url = item["url"]

    # iterate through section_content 
    for section, content in item["sections_content"].items():
        metadata = {
            "title": title,
            "url": url,
            "section": section,
        }
        docs.append(Document(text=content, metadata=metadata))

# Initialise semantic splitter
splitter = SemanticSplitterNodeParser(
    buffer_size=1,                       
    breakpoint_percentile_threshold=95,  
    embed_model=embed_model
)

# Chunking
final_nodes = []
for doc in docs:
    token_count = count_tokens(doc.text)
    
    if token_count > 512:  # LLMtoken阈值512
        nodes = splitter.get_nodes_from_documents([doc])
        final_nodes.extend(nodes)
    else:
        # 短section直接转为Node，不分割
        node = TextNode(text=doc.text, metadata=doc.metadata)
        final_nodes.append(node)

# Build index from final nodes
index = VectorStoreIndex(nodes=final_nodes,embed_model=embed_model)

# Store the index - just run it for the 1st time
# index.storage_context.persist(persist_dir="./storage_context_large") 

# Loading stored index
storage_context = StorageContext.from_defaults(persist_dir="./storage_context_large")
index = load_index_from_storage(storage_context)
