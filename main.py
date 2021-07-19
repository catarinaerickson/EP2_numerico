##############################################################################
#                               EP2 - MAP3121                                #
#       Autovalores e Autovetores de Matrizes Tridiagonais Simétricas        #
##############################################################################
#        Alunas:                                                             #
#                Catarina Rodrigues Erickson   (NUSP: 11258742)              #
#                Isabella Mulet e Souza        (NUSP: 11259184)              #
#                                                                            #
##############################################################################

#Importando bibliotecas para trabalhar com aritmética de vetores e gráficos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # Cabeçalho
    print("EP1 - MAP3121")
    print("Autovalores e Autovetores de Matrizes Tridiagonais Simétricas")
    print("Catarina Erickson e Isabella Mulet")
    print()   

    done = False
    while done == False:
        print()
        print("Tarefas")
        print()

        # Lista itens do enunciado
        print("a) Teste matriz simétrica 4x4 (input-a)")
        print()
        print('b) Teste matriz simétrica 20x20 (input-b)')
        print()
        print('c) Teste treliças planas (input-c)')
        print()

        # Lê o item selecionado
        check = True
        while(check == True):
            # Repete até ter uma entrada válida
            item = (input("Qual item deseja selecionar (a/b/c)?"))
            if item == "a" or item == "b" or item == "c" :
                # Verifica se a entrada é valida
                check = False
                print()
        
        # Item a
        if item == 'a':
            mat_A, autovalores, autovetores = test_qr('a')
            print(f'Matriz de entrada: \n {mat_A}')
            print(f'Autovalores (diagonal da matriz de autovalores): \n {autovalores}')
            print(f'Matriz de autovetores (contendo os autovetores em suas colunas): \n {autovetores}')
            print()
            
            check = True
            while(check == True):
                # Repete até ter uma entrada válida
                next = (input("Deseja verificar se Av=\u03BBv para cada autovalor e seu autovetor correpondente (s/n)?"))
                if next == 's' or next == 'n':
                    # Verifica se a entrada é valida
                    check = False
                    print()
            
            if next == 's':
                for i in range(mat_A.shape[0]):
                    A_v, lmb_v, igual = test_autovalor(mat_A, autovalores, autovetores, i)
                    print(f'Para o autovalor {autovalores[i]}, temos: \n Av = {A_v} \n \u03BBv = {lmb_v}')
                    if igual:
                        print('Com tolerância de 0.000001, podemos afirmar que Av = \u03BBv')
                    print()

            check = True
            while(check == True):
                # Repete até ter uma entrada válida
                next = (input("Deseja verificar a matriz de autovetores é ortogonal(s/n)?"))
                if next == 's' or next == 'n':
                    # Verifica se a entrada é valida
                    check = False
                    print()
            
            if next == 's':
                mat_id, igual_id = test_ort(autovetores)
                print(f'Produto entre a matriz de autovetores e sua transposta (deve ser igual à matriz identidade): \n {mat_id}')
                print()
                if igual_id:
                    print('Com tolerância de 0.000001, podemos afirmar que a matriz de autovetores é ortogonal')
                    print()

        # Item b  
        if item == 'b':
            mat_A, autovalores, autovetores = test_qr('b')
            print(f'Matriz de entrada: \n {mat_A}')
            print(f'Autovalores (diagonal da matriz de autovalores): \n {autovalores}')
            print(f'Matriz de autovetores (contendo os autovetores em suas colunas): \n {autovetores}')
            print()
            
            check = True
            while(check == True):
                # Repete até ter uma entrada válida
                next = (input("Deseja verificar se Av=\u03BBv para cada autovalor e seu autovetor correpondente (s/n)?"))
                if next == 's' or next == 'n':
                    # Verifica se a entrada é valida
                    check = False
                    print()
            
            if next == 's':
                for i in range(mat_A.shape[0]):
                    A_v, lmb_v, igual = test_autovalor(mat_A, autovalores, autovetores, i)                    
                    print(f'Para o autovalor {autovalores[i]}, temos: \n Av = {A_v} \n \u03BBv = {lmb_v}')
                    if igual:
                        print('Com tolerância de 0.000001, podemos afirmar que Av = \u03BBv')
                    print()

            check = True
            while(check == True):
                # Repete até ter uma entrada válida
                next = (input("Deseja verificar a matriz de autovetores é ortogonal(s/n)?"))
                if next == 's' or next == 'n':
                    # Verifica se a entrada é valida
                    check = False
                    print()
            
            if next == 's':
                mat_id, igual_id = test_ort(autovetores)
                print(f'Produto entre a matriz de autovetores e sua transposta (deve ser igual à matriz identidade): \n {mat_id}')
                print()
                if igual_id:
                    print('Com tolerância de 0.000001, podemos afirmar que a matriz de autovetores é ortogonal')
                    print()

        # Item c
        if item == 'c':
            v_freq, mat_z = test_trl()
            for i in range(v_freq.shape[0]):
                print(f'Para a frequência {v_freq[i]}, temos o modo de vibração {mat_z[i]}')
                print()

        # Verifica a necessidade de uma nova execução
        check = True
        while(check == True):
            # Repete até ter uma entrada válida
            repeat = (input("Deseja excecutar novamente (s/n)? "))
            if repeat == 's' or repeat == 'n':
                # Verifica se a entrada é valida
                check = False
                print()
        if repeat == 'n':
            done = True

# Realiza o teste para o input-c
# Retorna um vetor contendo as cinco menores frequências 
# e uma matriz contendo os respectivos modos de vibração em suas linhas
def test_trl():
    n_total, n, b, dens, area, md_elas, mat_bij, v_theta, v_l = read_input_trl('arquivos_de_entrada/input-c')

    mat_K, dg_M =  create_K_M(n_total, n, b, dens, area, md_elas, mat_bij, v_theta, v_l)
    mat_K_sim = calc_K_sim(mat_K, dg_M)

    mat_T, mat_Ht = create_tridiagonal(2*n, mat_K_sim)
    principal, sub = select_dgs(mat_T)
    v_autovalor, mat_HtV, k = qr(principal, sub, True, mat_Ht)
    freq_v, mat_z = select_freq(v_autovalor, mat_HtV, dg_M)

    return freq_v, mat_z

# Calcula e seleciona as cinco menores frequências e seus respectivos modos de vibração 
# a partir do vetor de autovalores, da matriz de autovetores e da diagonal da matriz M
def select_freq(v_autovalor, mat_autovetor, dg_M):
    # Toma a transposta da matriz de autovetores para que cada autovetor corresponda a uma linha da matriz
    mat_autovetor_t = transpose(mat_autovetor)

    # Compõe uma lista de listas que contém na primeira posição o autovalor e na segunda o autovetor correspondente
    lmb_v_unsorted = []
    for i in range(v_autovalor.shape[0]):
        lmb_v_unsorted.append([v_autovalor[i], mat_autovetor_t[i]])
    # Ordena a lista de listas pelos autovalores em ordem crescente
    lmb_v_sorted = sorted(lmb_v_unsorted, key = lambda x: x[0])

    # Inicializa a matriz dos 5 modos de vibração e o vetor com as suas cinco menores frequências
    mat_z = np.zeros((5, mat_autovetor.shape[0]))
    freq_v = np.array([])
    for k in range(5):
        # Calcula a frequência k a partir do autovalor k (freq = raiz do autovalor)
        freq = lmb_v_sorted[k][0]**(1/2)
        freq_v = np.append(freq_v, freq)

        # Calcula o vetor k do modo de vibração multiplicando cada valor i da diagonal da matriz M(-1/2) 
        # por cada valor i do autovetor (multiplicação de M(-1/2) à direita)
        for i in range(dg_M.shape[0]):
            lmb_v_sorted[k][1][i] = (1/(dg_M[i]**(1/2)))*lmb_v_sorted[k][1][i]

        # Atribui o vetor do modo de vibração k à linha k da matriz de modos de vibração
        mat_z[k] = lmb_v_sorted[k][1]
        
    return freq_v, mat_z

# Calcula a matriz K simétrica a partir da matriz de rigidez total e a diagonal da matriz M
def calc_K_sim(mat_K, dg_M):
    # Multiplica a matriz M(-1/2) à direita da matriz K 
    for i in range(mat_K.shape[0]):
        # Para cada valor i do vetor da diagonal de M(-1/2), multiplica esse valor pela linha i de K
        mat_K[i] = (1/((dg_M[i])**(1/2)))*mat_K[i]
        
    # Multiplica a matriz M(-1/2) à esquerda da matriz K
    mat_K_t = transpose(mat_K)
    for j in range(mat_K.shape[0]):
        # Para cada valor j do vetor da diagonal de M(-1/2), multiplica esse valor pela coluna j de K
        mat_K_t[j] = (1/((dg_M[j])**(1/2)))*mat_K_t[j]
    mat_K = transpose(mat_K_t)

    return mat_K

# Cria a matriz de rigidez total K e a matriz de massa M a partir dos dados de entrada do input-c
# Recebe o número total de nós, o número de nós soltos, o número de barras, 
# a densidade, a área de seção transversal, o módulo do elasticidade,
# a matriz que identifica os nós extremos de cada barra, um vetor com os ângulos de cada barra 
# e um com o comprimento de cada barra.
# Retorna a matriz K e a diagonal da matriz M
def create_K_M(n_total, n, b, dens, area, md_elas, mat_bij, v_theta, v_l):
    # Inicializa a matriz K com dimensão 2n e zeros em todas entradas
    mat_K = np.zeros((2*n_total,2*n_total))
    # Incializa o vetor da diagonal da matriz M com zeros em todas as entradas
    dg_M = np.zeros((2*n_total,))

    # Realiza uma iteração para cada barra da treliça
    for bar in range(b):
        # Calcula o seno e o cosseno dado o ângulo da barra
        cos = np.cos(v_theta[bar]*np.pi/180)
        sin = np.sin(v_theta[bar]*np.pi/180)

        # Identifica os nós extremos da barra
        i = mat_bij[bar][0]
        j = mat_bij[bar][1]        

        # Cria a matriz simétrica Kij
        lin_1 = np.array([cos**2, cos*sin, -(cos**2), -cos*sin])
        lin_2 = np.array([cos*sin, sin**2, -cos*sin, -(sin**2)])
        lin_3 = np.array([lin_1[2], lin_1[3], lin_1[0], lin_1[1]])
        lin_4 = np.array([lin_2[2], lin_2[3], lin_2[0], lin_2[1]])
        mat_Kij = (area*md_elas/v_l[bar])*np.array([lin_1, lin_2, lin_3, lin_4])

        # Estabelece as posições a serem adicionados os elementos da matriz Kij na matriz K
        pos = [2*i-2, 2*i-1, 2*j-2, 2*j-1]

        # Soma as contribuições de Kij em suas respectivas posições na matriz K
        lin_Kij = 0 # contador da linha de Kij
        for lin_K in pos:
            col_Kij = 0 # contador da coluna de Kij
            for col_K in pos:
                mat_K[lin_K][col_K] += mat_Kij[lin_Kij][col_Kij]
                col_Kij += 1
            lin_Kij += 1

        # Soma as contribuições da massa da barra no vetor da diagonal da matriz M
        for pos_M in pos:
            dg_M[pos_M] += (1/2)*dens*area*v_l[bar] 
        
    # Seleciona a matriz K e a diagonal M apenas para os nós ativos
    mat_K = mat_K[0:2*n,0:2*n]
    dg_M = dg_M[0:2*n]

    return mat_K, dg_M

# Realiza os testes para o input-a ou o input-b
# Recebe 'a' ou 'b' para identificar qual arquivo de entrada de ser lido
# Retorna a matriz de entrada A, o vetor de autovalores e a matriz de autovetores correspondente
def test_qr(caso):
    if (caso == 'a'):
        n, mat_A = read_input_mat('arquivos_de_entrada/input-a')
    else:
        n, mat_A= read_input_mat('arquivos_de_entrada/input-b')

    mat_T, mat_Ht = create_tridiagonal(n, mat_A)
    principal, sub = select_dgs(mat_T)
    v_autovalor, mat_HtV, k = qr(principal, sub, True, mat_Ht)

    return mat_A, v_autovalor, mat_HtV

# Testa se o produto de uma matriz por um autovetor 
# é igual ao produto do autovetor e seu autovalor correspondente.
# Recebe a matriz A, o vetor com os autovalores, a matriz de autovetores 
# e o índice i do autovalor a ser testado.
# Retorna o produto da matriz e o autovetor i, o produto do autovetor i e o autovalor i 
# e True se esses produtos forem iguais e False se não forem.
def test_autovalor(mat_original, v_autovalor, mat_autovetores, i):
    # Seleciona o autovetor armazenado na coluna da matriz de autovetores
    mat_autovetores_t = transpose(mat_autovetores)
    autovetor = mat_autovetores_t[i]

    # Calcula o produto da matriz e o autovetor i
    A_v = np.dot(mat_original, autovetor)
    # Calcula o produto do autovetor i e o autovalor i
    lmb_v = v_autovalor[i]*autovetor

    # Compara os produtos calculados
    v_compare = np.isclose(A_v, lmb_v, atol = 0.000001)
    igual = True
    for x in v_compare:
        if not x:
            igual = False
            break
    
    return A_v, lmb_v, igual

# Testa se uma matriz é ortogonal
# Recebe uma matriz e retorna seu produto pela transposta e 
# True se esse produto for uma matriz identidade e False se não for
def test_ort(mat):
    # Calcula o produto da matriz por sua transposta
    mat_t = transpose(mat)
    mat_id = np.dot(mat, mat_t)

    # Cria uma matriz identidade com a mesma dimensão da matriz de entrada
    id = create_id(mat.shape[0])

    # Compara o produto e a matriz identidade
    mat_compare = np.isclose(mat_id, id, atol=0.000001)
    igual = True
    for lin in mat_compare:
        for col in lin:
            if not col:
                igual = False
                break

    return mat_id, igual

# Recebe um matriz simétrica tridiagonal e 
# retorna um vetor com a diagonal principal e um vetor com a subdiagonal
def select_dgs(mat):
    principal = np.array([])
    sub = np.array([])
    for i in range(mat.shape[0]):
        principal = np.append(principal,mat[i][i])
        if i<mat.shape[0]-1:
            sub = np.append(sub, mat[i+1][i])
    return principal, sub

# Recebe a dimensão n e uma matriz simétrica e retorna a matriz T tridiagonal e Ht
def create_tridiagonal(n, mat_A):
    # Inicializa Ht como a matriz identidade
    mat_Ht = create_id(n)

    # Inicia o loop de n-2 iterações 
    for m in range(n-2):
        # Calcula o vetor w de cada iteração m
        a = select_col(mat_A, m, m)
        w = calc_w(a)

        # Calcula Hw*A alterando as n-m linhas da coluna m em diante
        mat_HA_t = transpose(mat_A.copy())
        for i in range(m, n):
            small_col = select_col(mat_A, m, i)
            small_col = mult_H(w, small_col)

            # Adiciona na coluna calculada os valores que não sofreram alteração
            col = np.array([])
            for k in range(m+1):
                col = np.append(col, mat_A[k][i])
            col = np.append(col, small_col)

            # Armazena a coluna m que será simétrica em HAH
            if i == m:
                lin_sim = col.copy()

            # Modifica a coluna calculada na matriz 
            mat_HA_t[i] = col        
        mat_HA = transpose(mat_HA_t)
        # Atribui o valor da linha simétrica ao fim da multiplicação Hw*A
        mat_HA[m] = lin_sim


        # Calcula Hw*A*Hw alterando as n-m colunas da linha m+1 em diante
        mat_HAH = mat_HA.copy()
        for i in range(m+1, n):
            small_lin = select_col(transpose(mat_HA), m, i)
            small_lin = mult_H(w, small_lin)

            # Adiciona na linha calculada os valores que não sofreram alteração
            lin = np.array([])
            for k in range(m+1):
                lin = np.append(lin, mat_HA[k][i])
            lin = np.append(lin, small_lin)

            # Modifica a linha calculada na matriz 
            mat_HAH[i] = lin

        # Calcula a matriz Ht=I*Hw1*Hw2...*Hwm
        new_mat_Ht = mat_Ht.copy()
        for i in range(1, n):
            small_lin_Ht = select_col(transpose(mat_Ht), m, i)
            small_lin_Ht = mult_H(w, small_lin_Ht)

            # Adiciona na linha calculada os valores que não sofreram alteração
            lin_Ht = np.array([])
            for k in range(m+1):
                lin_Ht = np.append(lin_Ht, mat_Ht[i][k])
            lin_Ht = np.append(lin_Ht, small_lin_Ht)

            # Modifica a linha calculada na matriz 
            new_mat_Ht[i] = lin_Ht 

        # Armazena as matrizes calculadas para a próxima iteração
        mat_A = mat_HAH.copy()
        mat_Ht = new_mat_Ht.copy()

    return  mat_A, mat_Ht

# Recebe o vetor wi da matriz Hwi da transformação de Householder e um vetor a e retorna Hwi*x
def mult_H(w, a):      
    product = a -2*(calc_vector_dot(w, a)/calc_vector_dot(w, w))*w
    return product

# Recebe a primeira coluna da submatriz e retorna o vetor wi da matriz Hwi da transformação de Householder
def calc_w(a):
    # Calcula a norma de a
    norm_a = calc_vector_dot(a, a)**(1/2)

    # Define o vetor ei
    e = np.zeros(a.shape)
    e[0]=1

    # Calcula wi
    w = a + (a[0]/abs(a[0]))*norm_a*e

    return w

# Recebe uma matriz e seleciona a coluna i a partir da linha itr+1 
def select_col(mat, itr, i):
    col = np.array([])
    for k in range(itr+1, mat.shape[0]):
        col = np.append(col, mat[k][i])
    return col

# Calcula o produto escalar entre dois vetores
def calc_vector_dot(v1, v2):
    product = 0
    for i in range(v1.shape[0]):
        product += v1[i]*v2[i]
    return product

# Recebe uma matriz e retorna sua transposta
def transpose(mat):
    lin = mat.shape[0]
    col = mat.shape[1]

    mat_t = np.zeros((col,lin))

    for i in range(lin):
        for j in range(col):
            mat_t[j][i] = mat[i][j]

    return mat_t

# Cria uma matriz identidade de dimensão n
def create_id(n):
    mat = np.zeros((n,n), dtype = float)
    for i in range(n):
        mat[i][i] = 1
    return mat

# Recebe o caminho da pasta para o input-a ou input-b e retorna a dimensão n e a matriz A
def read_input_mat(filepath):
    # Lê o arquivo de input
    file = open(filepath)
    input = file.readlines()

    # Armazena a dimensão da matriz
    n = int(input[0])

    # Armazena a matriz A
    mat = np.zeros((n,n))
    for i in range(1, n+1):
        lin_str = input[i].split()
        for j in range(n):
            lin_str[j] = float(lin_str[j].rstrip('\n'))
        lin = np.array(lin_str)
        mat[i-1] = lin

    return n, mat

# Recebe o caminho da pasta para o inpu-c e retorna o número total de nós, o número de nós soltos, 
# o número de barras, a densidade, a área de seção transversal, o módulo do elasticidade,
# a matriz que identifica os nós extremos de cada barra, um vetor com os ângulos de cada barra 
# e um com o comprimento de cada barra.
def read_input_trl(filepath):
    # Lê o arquivo de input
    file = open(filepath)
    input = file.readlines()

    # Da linha 1 do arquivo, armazena o número total de nós, o número de nós soltos e o número de barras
    lin_1 = input[0].split()
    n_total = int(lin_1[0])
    n_solto = int(lin_1[1])
    b_total = int(lin_1[2])

    # Da linha 2 do arquivo, armazena a densidade, a área de seção transversal e o módulo do elasticidade
    lin_2 = input[1].split()
    dens = float(lin_2[0])
    area = float(lin_2[1])
    md_elas = float(lin_2[2])*1000000000

    # Das demais linhas do arquivo, cria a matriz que identifica os nós extremos de cada barra(mat_bij), 
    # um vetor com os ângulos de cada barra (v_theta) e um com o comprimento de cada barra(v_l). 
    mat_bij = np.zeros((len(input)-2, 2)).astype(int)
    v_theta = np.array([])
    v_l = np.array([])
    for k in range(2, len(input)):
        lin = input[k].split()
        mat_bij[k-2] = np.array([int(lin[0]), int(lin[1])])
        v_theta = np.append(v_theta, float(lin[2]))
        v_l = np.append(v_l, float(lin[3]))

    return n_total, n_solto, b_total, dens, area, md_elas, mat_bij, v_theta, v_l

def bonus(mat_bij, v_theta, v_l, freq_v, matz_z):
    a=1

# Calcula u de acordo com a heurística de Wilkinson
def calc_heuristica(alpha_arr, beta_arr, n):
    d = (alpha_arr[n-1] - alpha_arr[n])/2
    if d>=0:
        sgn = 1
    else:
        sgn = -1
    u = alpha_arr[n] + d - sgn*((d**2 + beta_arr[n-1]**2)**(1/2))
    return u

# Calcula o seno e o cosseno da matriz Q de transformação de Givens
def calc_theta(alpha, beta):
    raiz = (alpha**2+beta**2)**(1/2)
    cos = alpha/raiz
    sen = -beta/raiz
    return (cos, sen)

# Algoritmo QR
# Recebe uma lista da matriz diagonal principal, uma lista da matriz subdiagonal,
# True ou False para o deslocamento espectral e a matriz V inicial
# Retorna os autovalores, a mariz de autovetores e o número de iterações
def qr(alpha_list, beta_list, deslocamento, v_arr):
    alpha_arr = np.array(alpha_list, dtype = float)
    beta_arr = np.array(beta_list, dtype = float)

    # Define a dimensão inicial da matriz Anxn
    dim = alpha_arr.shape[0]

    # Inicializa vetor para armazenar os autovalores encontrados
    autovalor_arr = np.array([])
    k=0
    # Itera para m = dim, dim-1, ..., 2
    for m in range(dim, 1, -1):
        
        # Itera enquanto o beta de referência da iteração m for maior do que 10e-6
        while abs(beta_arr[m-2])>=0.000001:
            # Calcula e subtrai ukI de A(k) a partir da iteração k caso haja deslocamento
            if k>0 and deslocamento:
                u = calc_heuristica(alpha_arr, beta_arr, m-1)
                u_arr = u*np.ones(alpha_arr.shape)
                alpha_arr = alpha_arr - u_arr

            # Inicializa os vetores para armazenar os valores de seno e cosseno da iteração k
            c_arr = np.array([])
            s_arr = np.array([])


            # Calcula a matriz R
            beta_fixed = beta_arr.copy()
            for i in range(m-1):
                # Calcula o seno e cosseno da iteração i e guarda nos respectivos vetores
                (cos, sen) = calc_theta(alpha_arr[i],beta_fixed[i])
                c_arr = np.append(c_arr, cos)
                s_arr = np.append(s_arr, sen)

                # Cria a diagonal principal e a sobrediagonal da matriz Qi*Qi-1*...*Q1*A
                alpha_R = alpha_arr.copy()
                beta_R = beta_arr.copy()

                # Modifica a linha i
                alpha_R[i] = cos*alpha_arr[i] - sen*beta_fixed[i]
                beta_R[i] = cos*beta_arr[i] - sen*alpha_arr[i+1]
                # Modifica a linha i+1
                alpha_R[i+1] = sen*beta_arr[i] + cos*alpha_arr[i+1]
                # Modifica a linha i+1 de beta apenas se não for a última iteração i=m-2
                if i < beta_arr.shape[0]-1:
                    beta_R[i+1] = cos*beta_arr[i+1]

                # Define as diagonais da matriz Qi*Qi-1*...*Q1*A como ponto de partida para a próxima iteração
                alpha_arr = alpha_R.copy()
                beta_arr = beta_R.copy()

            # Calcula A(k+1)
            for i in range(m-1):
                # Cria a diangonal principal e a sobrediagonal da matriz R*Q1T*...*QiT
                alpha_next = alpha_arr.copy()
                beta_next = beta_arr.copy()

                # Modifica a coluna i
                alpha_next[i] = c_arr[i]*alpha_arr[i] - s_arr[i]*beta_arr[i]
                # Modifica a coluna i+1
                beta_next[i] = -s_arr[i]*alpha_arr[i+1]
                alpha_next[i+1] = c_arr[i]*alpha_arr[i+1]

                # Define as diagonais da matriz R*Q1T*...*QiT como ponto de partida para a próxima iteração
                alpha_arr = alpha_next.copy()
                beta_arr = beta_next.copy()

            # Soma A(k+1) e ukI a partir da iteração k caso haja deslocamento
            if k>0 and deslocamento:
                alpha_arr = alpha_arr + u_arr 

            # Calcula V
            for i in range(m-1):
                v_next = v_arr.copy()
                # Altera a colunas i e i+1 para cada linha j da matriz V
                for j in range(dim):
                    v_next[j][i] = v_arr[j][i]*c_arr[i] - v_arr[j][i+1]*s_arr[i]
                    v_next[j][i+1] = v_arr[j][i]*s_arr[i] + v_arr[j][i+1]*c_arr[i]

                # Define as diagonais da matriz V*Q1T*...*QiT como ponto de partida para a próxima iteração
                v_arr = v_next.copy()
            k+=1   
        
        # Guarda o autovalor encontrado na iteração m no vetor de autovalores
        autovalor_arr = np.append(autovalor_arr, alpha_arr[m-1])

        # Define as diagonais da submatriz para a iteração m-1
        alpha_arr = np.delete(alpha_arr, m-1)
        beta_arr = np.delete(beta_arr, m-2)

    # Guarda o autovetor que sobrou depois da última iteração m-1
    autovalor_arr = np.append(autovalor_arr, alpha_arr[0])

    # Inverte o vetor dos autovalores para que o autovalor da posição i corresponda ao autovetor da coluna i da matriz V
    autovalor_arr = np.array(list(reversed(autovalor_arr)))

    # Retorna o vetor com os autovalores, a matriz V e o número de iterações, nessa ordem
    return autovalor_arr, v_arr, k

#Chama a função main()
if __name__ == "__main__":
    main()

