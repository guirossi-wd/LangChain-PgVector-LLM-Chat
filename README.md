# Desafio MBA Engenharia de Software com IA - Full Cycle

Como executar a solução, quais comandos usar 

0 - Configurar para projeto rodar

python3 -m venv venv  # 1ª vez apenas


1 - Activate #Sempre ao iniciar o dia

C:\Python\LangChain\venv\Scripts\Activate.ps1

1.1 - Atualizar requirements sempre que tiver atualizar de libs

pip install langchain langchain-openai langchain-google-genai python-dotenv beautifulsoup4 pypdf

pip install langchain_community

pip install langchain_postgres

pip freeze > requirements.txt

2 - Subir docker

docker-compose up -d


   2.1 - Outros comandos docker necessários

       docker-compose down -v     #fecha o docker / derruba

       docker exec -it postgres_rag psql -U postgres -d   rag -c "SELECT version();"  #valida se deu certo o docker com banco de dados


3 - python src/ingest.py

4 - python src/chat.py


Obs sobre .env
 - Foi utilizado GEMINI
 - o PDF está na raiz do projeto com o nome document.pdf
 - O arquivo docker-compose está configurando a porta 5433 para não correr o risco de dar conflito com postgres instalado na maquina local que tem a porta padrão 5432
 