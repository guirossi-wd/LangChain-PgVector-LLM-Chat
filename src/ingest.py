import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
load_dotenv()

#validação se ambiente está configurado
for k in ("GOOGLE_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Variável de Ambiente {k} não está disponível.")

docs = PyPDFLoader(os.getenv("PDF_PATH")).load()

def ingest_pdf():
    #chunk_size -> está dividindo a pagina em 1000 caracteres
    #chunk_overlap -> está aumentando em 150 caracteres antes do chunk para não perder contexto
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False
        ).split_documents(docs)
    if not splits:
        raise SystemExit(0)    

    #vejo numero de chunks que vai gerar... pq api do gemini free só permite 100 requests por 60seg
    print(f"\nTotal de chunks: {len(splits)}\n\n")


    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in splits
    ]    

    #pega a lista de ids de documentos enriquecidos
    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_MODEL", "models/gemini-embedding-001"))

    #inserir os bancos no DB usando o PGVector da LangChain (automágico)
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True
    )

    #aqui é o envio direto, sem limitação, porém com API free não é possível, dá erro :)
    #store.add_documents(documents=enriched, ids=ids)

    #aqui é inserção de 30 por minuto para não estourar permissão do gemini (recurso alternativo p/ funcionar)
    batch_size = 30
    for i in range(0, len(enriched), batch_size):
        print("-"*30)
        print(f"\nIniciando Ciclo em {i}")
        batch = enriched[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        store.add_documents(documents=batch, ids=batch_ids)
        print(f"\nLote {i//batch_size + 1} inserido\n")
        if i + batch_size < len(enriched):
            time.sleep(65)    


    print("="*50)
    print("Processo Finalizado com sucesso !")
    print("="*50)





if __name__ == "__main__":
    ingest_pdf()