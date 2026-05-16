from search import search_prompt    #estou importando a função do search
from dotenv import load_dotenv
load_dotenv()

def main():
    print("Chat iniciado. Digite 'sair', '0' ou deixe em branco para encerrar.\n")

    while True:
        pergunta = input("Digite sua pergunta: ").strip()

        if pergunta in ("", "0", "sair"):
            print("Encerrando o chat. Até logo!")
            break

        chain = search_prompt(pergunta)

        if not chain:
            print("Não foi possível processar a pergunta. Verifique os erros de inicialização.")
            continue

        print(f"\n{chain}\n")

if __name__ == "__main__":
    main()