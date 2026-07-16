import cv2
import numpy as np

def panoramica(img1_color, img2_color, metodo="SIFT", limiar_ratio=0.75):

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Encontrar pontos de interesse e descritores
    if metodo == "SIFT":
        detector = cv2.SIFT_create()
        norma = cv2.NORM_L2
    elif metodo == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
        norma = cv2.NORM_HAMMING
    elif metodo == "BRISK":
        detector = cv2.BRISK_create()
        norma = cv2.NORM_HAMMING
    else:
        print(f"Erro: Método '{metodo}' não suportado.")
        return None, None

    keypoints1, descriptors1 = detector.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2_gray, None)

    matcher = cv2.BFMatcher(norma, crossCheck=False)
    matches_brutos = matcher.knnMatch(descriptors1, descriptors2, k=2)

    bons_matches = []
    for m, n in matches_brutos:
        # Se a distância do melhor match for muito menor que a do segundo melhor (limiar)
        if m.distance < limiar_ratio * n.distance:
            bons_matches.append(m)

    # RANSAC exige pelo menos 4 pontos para calcular a matriz 3x3
    if len(bons_matches) >= 4:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)

        # 4. Encontrar a Homografia 
        H, mascara_ransac = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        altura1, largura1 = img1_color.shape[:2]
        altura2, largura2 = img2_color.shape[:2]
        
        # Cria uma tela grande para caber as duas imagens lado a lado
        largura_panorama = largura1 + largura2
        altura_panorama = max(altura1, altura2)
        
        # Distorce a Imagem 1 para o plano da Imagem 2
        img_panorama = cv2.warpPerspective(img1_color, H, (largura_panorama, altura_panorama))

        # 7. Unir as imagens
        # Substitui os pixels vazios à esquerda pela Imagem 2
        img_panorama[0:altura2, 0:largura2] = img2_color

        # 8. Desenhar as retas entre pontos correspondentes
        # Apenas os pontos que o RANSAC considerou como válidos
        matches_ransac = [bons_matches[i] for i in range(len(bons_matches)) if mascara_ransac[i]]
        
        img_linhas = cv2.drawMatches(
        img1_color, keypoints1,
        img2_color, keypoints2,
        matches_ransac,  # sem limite
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

        return img_panorama, img_linhas
    
    else:
        print(f"Erro: Foram encontradas apenas {len(bons_matches)} boas correspondências (mínimo 4).")
        return None, None
    
