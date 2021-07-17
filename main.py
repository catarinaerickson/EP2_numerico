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
        print("a) Teste matriz simétrica 4x4")
        print()
        print('b) Teste matriz simétrica 20x20')
        print()
        print('c) Teste treliças planas')
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
                    A_v, lmb_v = test_autovalor(mat_A, autovalores, autovetores, i)
                    print(f'Para o autovalor {autovalores[i]}, temos: \n Av = {A_v} \n \u03BBv = {lmb_v}')
                    print()

            check = True
            while(check == True):
                # Repete até ter uma entrada válida
                next = (input("Deseja verificar a matriz de autovetores é ortogonal?(s/n)?"))
                if next == 's' or next == 'n':
                    # Verifica se a entrada é valida
                    check = False
                    print()
            
            if next == 's':
                mat_id = test_ort(autovetores)
                print(f'Produto entre a matriz de autovetores e sua transposta (deve ser igual à matriz identidade): \n {mat_id}')
                print()
                
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
                    A_v, lmb_v = test_autovalor(mat_A, autovalores, autovetores, i)
                    print(f'Para o autovalor {autovalores[i]}, temos: \n Av = {A_v} \n \u03BBv = {lmb_v}')
                    print()

            check = True
            while(check == True):
                # Repete até ter uma entrada válida
                next = (input("Deseja verificar a matriz de autovetores é ortogonal?(s/n)?"))
                if next == 's' or next == 'n':
                    # Verifica se a entrada é valida
                    check = False
                    print()
            
            if next == 's':
                mat_id = test_ort(autovetores)
                print(f'Produto entre a matriz de autovetores e sua transposta (deve ser igual à matriz identidade): \n {mat_id}')
                print()


        # Verifica a necesssidade de uma nova excecução
        sel = input("Deseja excecutar novamente (s/n)? ")
        if sel == 'n':
            done = True

def test_qr(caso):
    if (caso == 'a'):
        n, mat_A = read_input_mat('arquivos_de_entrada/input-a')
    else:
        n, mat_A= read_input_mat('arquivos_de_entrada/input-b')
    mat_T, mat_Ht = create_tridiagonal(n, mat_A)
    principal, sub = select_dgs(mat_T)
    v_autovalor, mat_HtV, k = qr(principal, sub, True, mat_Ht)
    return mat_A, v_autovalor, mat_HtV

def test_autovalor(mat_original, v_autovalor, mat_autovetores, i):
    mat_autovetores_t = transpose(mat_autovetores)
    autovetor = mat_autovetores_t[i]
    A_v = np.dot(mat_original, autovetor)
    lmb_v = v_autovalor[i]*autovetor
    return A_v, lmb_v

def test_ort(mat):
    mat_t = transpose(mat)
    mat_id = np.dot(mat, mat_t)
    return mat_id

def select_dgs(mat):
    principal = np.array([])
    sub = np.array([])
    for i in range(mat.shape[0]):
        principal = np.append(principal,mat[i][i])
        if i<mat.shape[0]-1:
            sub = np.append(sub, mat[i+1][i])
    return principal, sub

def create_tridiagonal(n, mat_A):
    mat_Ht = create_id(n)
    for m in range(n-2):
        a = select_col(mat_A, m, m)
        w = calc_w(a)

        mat_HA_t = transpose(mat_A.copy())
        for i in range(m, n):
            small_col = select_col(mat_A, m, i)
            small_col = mult_H(w, small_col)
            col = np.array([])
            for k in range(m+1):
                col = np.append(col, mat_A[k][i])
            col = np.append(col, small_col)
            if i == m:
                lin_sim = col.copy()
            mat_HA_t[i] = col        
        mat_HA = transpose(mat_HA_t)
        mat_HA[m] = lin_sim


        mat_HAH = mat_HA.copy()
        for i in range(m+1, n):
            small_lin = select_col(transpose(mat_HA), m, i)
            small_lin = mult_H(w, small_lin)
            lin = np.array([])
            for k in range(m+1):
                lin = np.append(lin, mat_HA[k][i])
            lin = np.append(lin, small_lin)
            mat_HAH[i] = lin

        new_mat_Ht = mat_Ht.copy()
        for i in range(1, n):
            small_lin_Ht = select_col(transpose(mat_Ht), m, i)
            small_lin_Ht = mult_H(w, small_lin_Ht)
            lin_Ht = np.array([])
            for k in range(m+1):
                lin_Ht = np.append(lin_Ht, mat_Ht[i][k])
            lin_Ht = np.append(lin_Ht, small_lin_Ht)
            new_mat_Ht[i] = lin_Ht 

        mat_A = mat_HAH.copy()
        mat_Ht = new_mat_Ht.copy()

    return  mat_A, mat_Ht

def mult_H(w, a):      
    product = a -2*(calc_vector_dot(w, a)/calc_vector_dot(w, w))*w
    return product

def calc_w(a):
    norm_a = calc_vector_dot(a, a)**(1/2)
    e = np.zeros(a.shape)
    e[0]=1
    w = a + (a[0]/abs(a[0]))*norm_a*e
    return w

def select_col(mat, itr, i):
    col = np.array([])
    for k in range(itr+1, mat.shape[0]):
        col = np.append(col, mat[k][i])
    return col

def calc_vector_dot(v1, v2):
    product = 0
    for i in range(v1.shape[0]):
        product += v1[i]*v2[i]
    return product

def transpose(mat):
    lin = mat.shape[0]
    col = mat.shape[1]

    mat_t = np.zeros((col,lin))

    for i in range(lin):
        for j in range(col):
            mat_t[j][i] = mat[i][j]

    return mat_t

def create_id(n):
    mat = np.zeros((n,n), dtype = float)
    for i in range(n):
        mat[i][i] = 1
    return mat

def read_input_mat(filepath):
    file = open(filepath)
    input = file.readlines()
    n = int(input[0])
    mat = np.zeros((n,n))
    for i in range(1, n+1):
        lin_str = input[i].split()
        for j in range(n):
            lin_str[j] = float(lin_str[j].rstrip('\n'))
        lin = np.array(lin_str)
        mat[i-1] = lin

    return n, mat

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

