import argparse
import cv2
from transformacoes import interpolacao_vizinho_proximo, interpolacao_bilinear, interpolacao_bicubica, interpolacao_Lagrange

def main():
    # Inicializa o capturador de argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Trabalho 4 - PDI")
    
    # Adiciona as flags conforme o enunciado
    parser.add_argument('-a', type=float,default=0.0, help='Ângulo de rotação em graus')
    parser.add_argument('-e', type=float, default=1.0, help='Fator de escala')
    parser.add_argument('-d', nargs=2, type=int, metavar=('LARGURA', 'ALTURA'), help='Dimensão da imagem de saída')
    parser.add_argument('-m', type=str, help='Método de interpolação')
    parser.add_argument('-i', type=str, required=True, help='Imagem de entrada (PNG)')
    parser.add_argument('-o', type=str, required=True, help='Imagem de saída (PNG)')

    # Extrai os argumentos digitados no terminal
    args = parser.parse_args()

    # Exemplo de leitura da imagem (permitido usar OpenCV/NumPy apenas para isso na Parte 1)
    img_entrada = cv2.imread(args.i)
    
    if img_entrada is None:
        print(f"Erro: Não foi possível ler a imagem {args.i}")
        return

    print(f"Processando {args.i} para salvar em {args.o}...")

    largura_out = args.d[0] if args.d is not None else None
    altura_out = args.d[1] if args.d is not None else None
    
    # --- AQUI ENTRA SUA LÓGICA DE TRANSFORMAÇÃO GEOMÉTRICA OU REGISTRO ---
    if args.m == "vizinho_proximo":
        imagem_saida = interpolacao_vizinho_proximo(img_entrada, escala=args.e, angulo=args.a, largura_saida=largura_out, altura_saida=altura_out)
    elif args.m == "bilinear":
        imagem_saida = interpolacao_bilinear(img_entrada, escala=args.e, angulo=args.a, largura_saida=largura_out, altura_saida=altura_out)
    elif args.m == "bicubica":
        imagem_saida = interpolacao_bicubica(img_entrada, escala=args.e, angulo=args.a, largura_saida=largura_out, altura_saida=altura_out)
    elif args.m == "Lagrange":
        imagem_saida = interpolacao_Lagrange(img_entrada, escala=args.e, angulo=args.a, largura_saida=largura_out, altura_saida=altura_out)
    else:
        print(f"Método de interpolação '{args.m}' não reconhecido. Use 'vizinho_proximo', 'bilinear', 'bicubica' ou 'Lagrange'.")
        return
    
    cv2.imwrite(args.o, imagem_saida)
    print(f"Imagem salva em {args.o}")

if __name__ == "__main__":
    main()