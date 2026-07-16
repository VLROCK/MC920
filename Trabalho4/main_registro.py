import argparse
import cv2
import sys 
from panoramica import panoramica

def main():
    parser = argparse.ArgumentParser(description="Trabalho 4 - PDI: Registro de Imagens (Parte 2)")
    
    # Parâmetros esperados no terminal
    parser.add_argument('-i', nargs=2, type=str, required=True, metavar=('IMG1', 'IMG2'), help='Caminho para as duas imagens de entrada (ex: A.jpg B.jpg)')
    parser.add_argument('-o', type=str, required=True, help='Caminho para salvar a imagem panorâmica de saída')
    parser.add_argument('-m', type=str, default='SIFT', choices=['SIFT', 'ORB', 'BRISK'], help='Método de detecção de pontos de interesse (Padrão: SIFT)')
    parser.add_argument('-l', type=float, default=0.75, help='Limiar para o Teste de Ratio de Lowe (Padrão: 0.75)')

    args = parser.parse_args()

    print(f"Carregando as imagens: {args.i[0]} e {args.i[1]}...")
    img1 = cv2.imread(args.i[0])
    img2 = cv2.imread(args.i[1])

    if img1 is None:
        print(f"Erro ao carregar a imagem 1: {args.i[0]}")
        sys.exit(1)
    if img2 is None:
        print(f"Erro ao carregar a imagem 2: {args.i[1]}")
        sys.exit(1)

    print(f"Iniciando o registro usando o método {args.m}...")
    img_panorama, img_linhas = panoramica(img1, img2, metodo=args.m, limiar_ratio=args.l)

    if img_panorama is not None:
        # Salva a imagem panorâmica final
        cv2.imwrite(args.o, img_panorama)
        print(f"Sucesso! Imagem panorâmica guardada em: {args.o}")
        
        # Salva a imagem auxiliar com as linhas de correspondência (exigida para o relatório)
        nome_linhas = args.o.replace(".jpg", "_linhas.jpg").replace(".png", "_linhas.png")
        cv2.imwrite(nome_linhas, img_linhas)
        print(f"Imagem com as linhas de correspondência guardada em: {nome_linhas}")
    else:
        print("Falha ao gerar a imagem panorâmica.")

if __name__ == "__main__":
    main()