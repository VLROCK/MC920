import numpy as np

def interpolacao_vizinho_proximo(imagem, escala=1.0, angulo=0.0,largura_saida=None, altura_saida=None):
    """
    Implementação da interpolação pelo vizinho mais próximo.
    """
    altura, largura, canais = imagem.shape

    # Calcula a escala em x e y com base nos parâmetros fornecidos
    escala_x = (largura_saida / largura) if largura_saida is not None else escala
    escala_y = (altura_saida / altura) if altura_saida is not None else escala

    nova_altura = int(altura * escala_y)
    nova_largura = int(largura * escala_x)

    # Cria uma imagem de saída com as novas dimensões
    imagem_saida = np.zeros((nova_altura, nova_largura, canais), dtype=imagem.dtype)

    y_out, x_out = np.indices((nova_altura, nova_largura))
    
   #Escala
    if escala_x != 1.0 or escala_y != 1.0:

        # Calcula as coordenadas correspondentes na imagem original usando o rounding para o vizinho mais próximo
        x_in = np.round(x_out / escala_x).astype(int)
        y_in = np.round(y_out / escala_y).astype(int)

        # Cria uma máscara para garantir que as coordenadas estejam dentro dos limites da imagem original.
        mascara = (x_in >= 0) & (x_in < largura) & (y_in >= 0) & (y_in < altura)


        imagem_saida[y_out[mascara], x_out[mascara]] = imagem[y_in[mascara], x_in[mascara]]
    else:
        imagem_saida = imagem.copy()

    #Rotação
    if angulo != 0.0:
        angulo_rad = np.radians(angulo)
        cos_angulo = np.cos(angulo_rad)
        sin_angulo = np.sin(angulo_rad)

        # Calcula as dimensões da imagem rotacionada para que caiba a imagem inteira após a rotação sem cortes.
        altura_rot = int(np.round(np.abs(cos_angulo) * nova_altura + np.abs(sin_angulo) * nova_largura)) + 1
        largura_rot = int(np.round(np.abs(cos_angulo) * nova_largura + np.abs(sin_angulo) * nova_altura)) + 1

        centro_x_nov = largura_rot / 2.0
        centro_y_nov = altura_rot / 2.0

        centro_x_ant = nova_largura / 2.0
        centro_y_ant = nova_altura / 2.0

        imagem_rotacionada = np.zeros((altura_rot, largura_rot, canais), dtype=imagem.dtype)

        # Calcula as coordenadas de saída para a imagem rotacionada
        y_out_rot, x_out_rot = np.indices((altura_rot, largura_rot))

        dx = x_out_rot - centro_x_nov
        dy = y_out_rot - centro_y_nov

        # Calcula as coordenadas correspondentes na imagem de saída escalada usando a rotação inversa
        x_in = np.round(dx*cos_angulo + dy*sin_angulo + centro_x_ant).astype(int)
        y_in = np.round(-dx*sin_angulo + dy*cos_angulo + centro_y_ant).astype(int)

        mascara = (x_in >= 0) & (x_in < nova_largura) & (y_in >= 0) & (y_in < nova_altura)

        imagem_rotacionada[y_out_rot[mascara], x_out_rot[mascara]] = imagem_saida[y_in[mascara], x_in[mascara]]


        imagem_saida = imagem_rotacionada

    
    return imagem_saida

def interpolacao_bilinear(imagem, escala=1.0, angulo=0.0, largura_saida=None, altura_saida=None):
    """
    Implementação da interpolação bilinear.
    """
    altura, largura, canais = imagem.shape

    escala_x = (largura_saida / largura) if largura_saida is not None else escala
    escala_y = (altura_saida / altura) if altura_saida is not None else escala

    nova_altura = int(altura * escala_y)
    nova_largura = int(largura * escala_x)
    imagem_saida = np.zeros((nova_altura, nova_largura, canais), dtype=imagem.dtype)

    y_out, x_out = np.indices((nova_altura, nova_largura))
    
   #Escala
    if escala_x != 1.0 or escala_y != 1.0:

        x_in = (x_out / escala_x)
        y_in = (y_out / escala_y)

        mascara = (x_in >= 0) & (x_in < largura) & (y_in >= 0) & (y_in < altura)

        x_in_round = np.floor(x_in).astype(int)
        y_in_round = np.floor(y_in).astype(int)

        dx = x_in - x_in_round
        dy = y_in - y_in_round

        dx_m = dx[mascara][..., np.newaxis]
        dy_m = dy[mascara][..., np.newaxis]

        x_base = x_in_round[mascara]
        y_base = y_in_round[mascara]

        x_prox = np.clip(x_base + 1, 0, largura - 1)
        y_prox = np.clip(y_base + 1, 0, altura - 1)

        f_x_y   = imagem[y_base, x_base].astype(float)
        f_x_y1  = imagem[y_prox, x_base].astype(float)
        f_x1_y  = imagem[y_base, x_prox].astype(float)
        f_x1_y1 = imagem[y_prox, x_prox].astype(float)

        composicao = (
            f_x_y * (1 - dx_m) * (1 - dy_m) +
            f_x_y1 * (1 - dx_m) * dy_m +
            f_x1_y * dx_m * (1 - dy_m) +
            f_x1_y1 * dx_m * dy_m
        )

        imagem_saida[y_out[mascara], x_out[mascara]] = composicao.astype(np.uint8)

    else:
        imagem_saida = imagem.copy()

    #Rotação
    if angulo != 0.0:
        angulo_rad = np.radians(angulo)
        cos_angulo = np.cos(angulo_rad)
        sin_angulo = np.sin(angulo_rad)

        altura_rot = int(np.round(np.abs(cos_angulo) * nova_altura + np.abs(sin_angulo) * nova_largura)) + 1
        largura_rot = int(np.round(np.abs(cos_angulo) * nova_largura + np.abs(sin_angulo) * nova_altura)) + 1

        centro_x_nov = largura_rot / 2.0
        centro_y_nov = altura_rot / 2.0

        centro_x_ant = nova_largura / 2.0
        centro_y_ant = nova_altura / 2.0

        imagem_rotacionada = np.zeros((altura_rot, largura_rot, canais), dtype=imagem.dtype)

        y_out_rot, x_out_rot = np.indices((altura_rot, largura_rot))

        dx_rot = x_out_rot - centro_x_nov
        dy_rot = y_out_rot - centro_y_nov

        x_in = dx_rot*cos_angulo + dy_rot*sin_angulo + centro_x_ant
        y_in = -dx_rot*sin_angulo + dy_rot*cos_angulo + centro_y_ant

        mascara = (x_in >= 0) & (x_in < nova_largura) & (y_in >= 0) & (y_in < nova_altura)

        x_in_round = np.floor(x_in).astype(int)
        y_in_round = np.floor(y_in).astype(int)

        dx = x_in - x_in_round
        dy = y_in - y_in_round

        dx_m = dx[mascara][..., np.newaxis]
        dy_m = dy[mascara][..., np.newaxis]

        x_base = x_in_round[mascara]
        y_base = y_in_round[mascara]

        # Correção do Index aplicada à rotação
        x_prox = np.clip(x_base + 1, 0, nova_largura - 1)
        y_prox = np.clip(y_base + 1, 0, nova_altura - 1)

        f_x_y   = imagem_saida[y_base, x_base].astype(float)
        f_x_y1  = imagem_saida[y_prox, x_base].astype(float)
        f_x1_y  = imagem_saida[y_base, x_prox].astype(float)
        f_x1_y1 = imagem_saida[y_prox, x_prox].astype(float)

        composicao = (
            f_x_y * (1 - dx_m) * (1 - dy_m) +
            f_x_y1 * (1 - dx_m) * dy_m +
            f_x1_y * dx_m * (1 - dy_m) +
            f_x1_y1 * dx_m * dy_m
        )

        imagem_rotacionada[y_out_rot[mascara], x_out_rot[mascara]] = composicao.astype(np.uint8)


        imagem_saida = imagem_rotacionada


    return imagem_saida


def P(t):
    return np.maximum(t, 0)


def R(s):
    return (1/6)*(P(s+2)**3 - 4*P(s+1)**3 + 6*P(s)**3 - 4*P(s-1)**3)


def _bicubica(img_in, img_out, x_float, y_float, x_out, y_out, limite_x, limite_y):
    mistura = np.zeros(img_out.shape, dtype=float)

    mascara = (x_float >= 0) & (x_float < limite_x) & (y_float >= 0) & (y_float < limite_y)

    x_in_round = np.floor(x_float).astype(int)
    y_in_round = np.floor(y_float).astype(int)

    # Calcula os deslocamentos relativos para a interpolação bicúbica
    dx = x_float - x_in_round
    dy = y_float - y_in_round

    # Calcula as coordenadas base para a interpolação bicúbica
    dx_m = dx[mascara][..., np.newaxis]
    dy_m = dy[mascara][..., np.newaxis]
    
    # Calcula as coordenadas base para a interpolação bicúbica
    x_base = x_in_round[mascara]
    y_base = y_in_round[mascara]

    # Itera sobre os vizinhos 4x4 ao redor do pixel de entrada
    for m in range(-1, 3):
        for n in range(-1, 3):
            x_vizinho = np.clip(x_base + m, 0, limite_x - 1)
            y_vizinho = np.clip(y_base + n, 0, limite_y - 1)

            peso_m = R(m - dx_m)
            peso_n = R(dy_m - n)
            
            # Calcula a cor do pixel vizinho e acumula na mistura ponderada
            cor_vizinho = img_in[y_vizinho, x_vizinho].astype(float)
            
            mistura[y_out[mascara], x_out[mascara]] += cor_vizinho * peso_m * peso_n

    mistura_clipada = np.clip(mistura[y_out[mascara], x_out[mascara]], 0, 255)
    img_out[y_out[mascara], x_out[mascara]] = mistura_clipada.astype(np.uint8)

    return img_out


def interpolacao_bicubica(imagem, escala=1.0, angulo=0.0, largura_saida=None, altura_saida=None):
    """
    Implementação da interpolação bicúbica.
    """
    altura, largura, canais = imagem.shape

    escala_x = (largura_saida / largura) if largura_saida is not None else escala
    escala_y = (altura_saida / altura) if altura_saida is not None else escala

    nova_altura = int(altura * escala_y)
    nova_largura = int(largura * escala_x
                       )
    imagem_saida = np.zeros((nova_altura, nova_largura, canais), dtype=imagem.dtype)

    y_out, x_out = np.indices((nova_altura, nova_largura))
    
   #Escala
    if escala_x != 1.0 or escala_y != 1.0:

        x_in = (x_out / escala_x)
        y_in = (y_out / escala_y)

        imagem_saida = _bicubica(imagem, imagem_saida, x_in, y_in, x_out, y_out, largura, altura)
      
    else:
        imagem_saida = imagem.copy()

    #Rotação
    if angulo != 0.0:
        angulo_rad = np.radians(angulo)
        cos_angulo = np.cos(angulo_rad)
        sin_angulo = np.sin(angulo_rad)

        altura_rot = int(np.round(np.abs(cos_angulo) * nova_altura + np.abs(sin_angulo) * nova_largura)) + 1
        largura_rot = int(np.round(np.abs(cos_angulo) * nova_largura + np.abs(sin_angulo) * nova_altura)) + 1

        centro_x_nov = largura_rot / 2.0
        centro_y_nov = altura_rot / 2.0

        centro_x_ant = nova_largura / 2.0
        centro_y_ant = nova_altura / 2.0

        imagem_rotacionada = np.zeros((altura_rot, largura_rot, canais), dtype=imagem.dtype)

        y_out_rot, x_out_rot = np.indices((altura_rot, largura_rot))

        dx = x_out_rot - centro_x_nov
        dy = y_out_rot - centro_y_nov

        x_in = (dx*cos_angulo + dy*sin_angulo + centro_x_ant)
        y_in = (-dx*sin_angulo + dy*cos_angulo + centro_y_ant)

        imagem_rotacionada = _bicubica(imagem_saida, imagem_rotacionada, x_in, y_in, x_out_rot, y_out_rot, nova_largura, nova_altura)


        imagem_saida = imagem_rotacionada


    return imagem_saida


def _calculo_Lagrange(img_in, img_out, x_float, y_float, x_out, y_out, limite_x, limite_y):

    mascara = (x_float >= 0) & (x_float < limite_x) & (y_float >= 0) & (y_float < limite_y)

    x_in_round = np.floor(x_float).astype(int)
    y_in_round = np.floor(y_float).astype(int)

    dx = x_float - x_in_round
    dy = y_float - y_in_round

    # Calcula os deslocamentos relativos para a interpolação de Lagrange
    dx_m = dx[mascara][..., np.newaxis]
    dy_m = dy[mascara][..., np.newaxis]
    
    # Calcula as coordenadas base para a interpolação de Lagrange
    x_base = x_in_round[mascara]
    y_base = y_in_round[mascara]

    # Função auxiliar para obter o valor do pixel na posição (ox, oy) relativa à base (x_base, y_base)
    def f(ox, oy):
        px = np.clip(x_base + ox, 0, limite_x - 1)
        py = np.clip(y_base + oy, 0, limite_y - 1)
        return img_in[py, px].astype(float)

    # Função auxiliar para calcular o polinômio de Lagrange
    def L(n):

        oy = n - 2 
        
        term1 = (-dx_m * (dx_m - 1) * (dx_m - 2) * f(-1, oy)) / 6.0
        term2 = ((dx_m + 1) * (dx_m - 1) * (dx_m - 2) * f(0, oy)) / 2.0
        term3 = (-dx_m * (dx_m + 1) * (dx_m - 2) * f(1, oy)) / 2.0
        term4 = (dx_m * (dx_m + 1) * (dx_m - 1) * f(2, oy)) / 6.0
        
        return term1 + term2 + term3 + term4

    L1 = L(1)
    L2 = L(2)
    L3 = L(3)
    L4 = L(4)


    mistura = (
        (-dy_m * (dy_m - 1) * (dy_m - 2) * L1) / 6.0 +
        ((dy_m + 1) * (dy_m - 1) * (dy_m - 2) * L2) / 2.0 +
        (-dy_m * (dy_m + 1) * (dy_m - 2) * L3) / 2.0 +
        (dy_m * (dy_m + 1) * (dy_m - 1) * L4) / 6.0
    )

    mistura_clipada = np.clip(mistura, 0, 255)
    img_out[y_out[mascara], x_out[mascara]] = mistura_clipada.astype(np.uint8)

    return img_out


def interpolacao_Lagrange(imagem, escala=1.0, angulo=0.0, largura_saida=None, altura_saida=None):
    """
    Implementação da interpolação de Lagrange.
    """
    altura, largura, canais = imagem.shape

    escala_x = (largura_saida / largura) if largura_saida is not None else escala
    escala_y = (altura_saida / altura) if altura_saida is not None else escala

    nova_altura = int(altura * escala_y)
    nova_largura = int(largura * escala_x)

    imagem_saida = np.zeros((nova_altura, nova_largura, canais), dtype=imagem.dtype)

    y_out, x_out = np.indices((nova_altura, nova_largura))
    
   #Escala
    if escala_x != 1.0 or escala_y != 1.0:

        x_in = (x_out / escala_x)
        y_in = (y_out / escala_y)

        imagem_saida = _calculo_Lagrange(imagem, imagem_saida, x_in, y_in, x_out, y_out, largura, altura)
    else:
        imagem_saida = imagem.copy()

    #Rotação
    if angulo != 0.0:
        angulo_rad = np.radians(angulo)
        cos_angulo = np.cos(angulo_rad)
        sin_angulo = np.sin(angulo_rad)

        altura_rot = int(np.round(np.abs(cos_angulo) * nova_altura + np.abs(sin_angulo) * nova_largura)) + 1
        largura_rot = int(np.round(np.abs(cos_angulo) * nova_largura + np.abs(sin_angulo) * nova_altura)) + 1

        centro_x_nov = largura_rot / 2.0
        centro_y_nov = altura_rot / 2.0

        centro_x_ant = nova_largura / 2.0
        centro_y_ant = nova_altura / 2.0

        imagem_rotacionada = np.zeros((altura_rot, largura_rot, canais), dtype=imagem.dtype)

        y_out_rot, x_out_rot = np.indices((altura_rot, largura_rot))

        dx = x_out_rot - centro_x_nov
        dy = y_out_rot - centro_y_nov

        x_in = (dx*cos_angulo + dy*sin_angulo + centro_x_ant)
        y_in = (-dx*sin_angulo + dy*cos_angulo + centro_y_ant)

        imagem_rotacionada = _calculo_Lagrange(imagem_saida, imagem_rotacionada, x_in, y_in, x_out_rot, y_out_rot, nova_largura, nova_altura)


        imagem_saida = imagem_rotacionada

    
    return imagem_saida