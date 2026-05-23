import os
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_postgres import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{resultados concatenados do banco de dados}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

@tool("pesquisa_banco_dados")
def pesquisa_banco_dados(query: str) -> str:
  """
    Função para fazer a pesquisa no banco de dados da pergunta do usuário.
  """

  for k in ("GOOGLE_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Variável de Ambiente {k} não está configurada.")

  #seleciona modelo
  embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_MODEL", "models/gemini-embedding-001"))

  store = PGVector(
      embeddings=embeddings,
      collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
      connection=os.getenv("DATABASE_URL"),
      use_jsonb=True
  )

  #pesquisa por similaridade
  results = store.similarity_search_with_score(query, k=10)
   
  textos = []

  for i, (doc, score) in enumerate(results, start=1):
        textos.append(
            f"[Documento {i} - score {score:.4f}]\n{doc.page_content.strip()}"
        )

  return "\n\n".join(textos)

def search_prompt(pergunta=None):    

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", disable_streaming=False)
  
    tools=[pesquisa_banco_dados]
  
    agent_chain = create_react_agent(
      llm,
      tools,
      prompt=SystemMessage(content=PROMPT_TEMPLATE)
    )

    result = agent_chain.invoke({"messages": [("human", pergunta)]})

    # O invoke retorna um dict com a chave "messages" contendo a lista de todas as
    # mensagens trocadas: HumanMessage, AIMessage (tool_call), ToolMessage e AIMessage (resposta final).
    # Pegamos sempre o último item, que é o AIMessage com a resposta final do agente.
    last_message = result["messages"][-1]

    # O Gemini às vezes retorna o content como lista de dicts no formato
    # [{'type': 'text', 'text': '...', 'extras': {...}}] em vez de uma string simples.
    # Nesse caso, extraímos apenas os blocos de texto e juntamos em uma única string.
    content = last_message.content
    if isinstance(content, list):
        content = " ".join(item["text"] for item in content if item.get("type") == "text")

    return content