# -*- coding: utf-8 -*-
def transpose(mat): #  "return transpose of mat"
    new_mat = zero(cols(mat),rows(mat))
    for row in range(rows(mat)):
        for col in range(cols(mat)):
            new_mat[col][row] = mat[row][col]
    return(new_mat)

def dot(A,B):
#    "vector dot product"
    if len(A) != len(B):
        print("dot: list lengths do not match")
        return()
    dot=0
    for i in range(len(A)):
        dot = dot + A[i]*B[i]
    return(dot)

def getCol(mat, col):
#    "return column col from matrix mat"
    return([r[col] for r in mat])

def getRow(mat, row):
#    "return row row from matrix mat"
    return(mat[row])

def matMult(mat1,mat2):
#    "multiply two matrices"
    if cols(mat1) != rows(mat2):
        print("matMult: mismatched matrices")
        return()
    prod = zero(rows(mat1),cols(mat2))
    for row in range(rows(mat1)):
        for col in range(cols(mat2)):
            prod[row][col] = dot(mat1[row],getCol(mat2,col))
    return(prod)

def vectorQ(V):
#    "mild test to see if V is a vector"
    if type(V) != type([1]):
        return(False)
    if type(V[0]) == type([1]):
        return(False)
    return(True)

def scalarMult(a,mat):
#    "multiply a scalar times a matrix"
    if vectorQ(mat):
        return([a*m for m in mat])
    for row in range(rows(mat)):
        for col in range(cols(mat)):
            mat[row][col] = a*mat[row][col]
    return(mat)

def addVectors(A,B):
 #   "add two vectors"
    if len(A) != len(B):
        print("addVectors: different lengths")
        return()
    return([A[i]+B[i] for i in range(len(A))])

def copyMatrix(M):
    return([[M[row][col] for col in range(cols(M))]for row in range(rows(M))])

def addrows(M, f, t, scale=1):
#    "add scale times row f to row t"
    N=copyMatrix(M)
    T=addVectors(scalarMult(scale,N[f]),N[t])
    N[t]=T
    return(N)
def zero(n , m):
#    "Create zero matrix"
    new_mat = [[0.0 for col in range(n)] for row in range(m)]
    return new_mat

def rows(mat):
#   "return number of rows"
    return(len(mat))

def cols(mat):
#    "return number of cols"
    return(len(mat[0]))

def augment(mat,vec):
#    "given nxn mat and n length vector return augmented matrix"
    amat = []
    show(amat)
    for row in range(rows(mat)):
        amat.append(mat[row]+[vec[row]])
    return(amat)

def createMatrix(A, n):
    print("Inside CreateMatrix")
    for i in range(0, n):
        for j in range(0,rows(A)):
            while True:
               element = input("Enter Number: ")
        if element > 0:
          A[i][j] = element
          break
    return(A)

def createVect(n):
  return([0.0 for i in range(n)])

def show(mat):
#    "Print out matrix"
    for row in mat:
        print(row)
        
def swaprows(M,i,j):
#    "swap rows i and j in matrix M"
    N=copyMatrix(M)
    T = N[i]
    N[i] = N[j]
    N[j] = T
    return N

def copyVect( b ) :
     return ([b[i] for i in range(len(b))])
    
def subVectors(A,B):
    if len(A) != len(B):
        print("SubVectors: different lengths")
        return()
    return([A[i]-B[i] for i in range(len(A))])

def multMat(mat1,mat2):
    "multiply two matrices"
    result = [[]]
    result = [[sum(i*j for i,j in zip(mat1Row, mat2Col)) for mat2Col in zip(*mat2)] for mat1Row in mat1]
    return(result)

def findPivotrow1(mat,col):
#    Finds index of the first row with nonzero entry on or
#    below diagonal.  If there isn't one return(-1).

    epsilon = 10**(-17)
    for row in range(col, rows(mat)):
#        if mat[row][col] != 0:
        if abs(mat[row][col]) > epsilon:
            return(row)
    return(-1)

def rowReduce(M):
#    return row reduced version of M
    N = copyMatrix(M)
    cs = cols(M)-2   # no need to consider last two cols
    rs = rows(M)
    for col in range(cs+1):
        j = findPivotrow1(N,col)
        if j < 0:
            print("\nrowReduce: No pivot found for column index %d "%(col))
            return(N)
        else:
            if j != col:
                N = swaprows(N,col,j)
            scale = -1.0 / N[col][col]
            for row in range(col+1,rs):                
                N=addrows(N, col, row, scale * N[row][col])
    print('rowReduce')
    show(N)
    return(N)


def backSub(M):

#   given a row reduced augmented matrix with nonzero 
#   diagonal entries, returns a solution vector
    

    cs = cols(M)-1 # cols not counting augmented col
    sol = [0.0 for i in range(cs)] # place for solution vector
    for i in range(1,cs+1):
        row = cs-i # work backwards
        sol[row] = ((M[row][cs] - sum([M[row][j]*sol[j] for
                    j in range(row+1,cs)])) / M[row][row]) 
    return(sol)


def diag_test(mat):

#   Returns True if no diagonal element is zero, False
#   otherwise.
    

    for row in range(rows(mat)):
        if mat[row][row] == 0:
            return(False)
    else:
        return(True)

def ge_1(aug):    

#   Given an augmented matrix it returns a list.  The [0]
#   element is the row-reduced augmented matrix, and 
#   ge_1(aug)[1] is a solution vector.  The solution
#   vector is empty if there is no unique solution.
    
    aug_n = rowReduce(aug)
    if diag_test(aug_n):
        sol = backSub(aug_n)
    else:
        print("\nge_1(): There is no unique solution")
        sol = []
    results = [aug_n, sol]
    return(results)


### The next two functions support checking a solution.

def getAandb(aug):
#   Returns the coef. matrix A and the vector b of Ax=b
    m = rows(aug)
    n = cols(aug)
    A = zero(m,n-1)
    b = zero(m,1)
    for i in range(m):
        for j in range(n-1):
            A[i][j] = aug[i][j]
    for i in range(m):
        b[i] = aug[i][n-1]
    Aandb = [A,b]
    return(Aandb)

def checkSol_1(aug,x):
#   For aug=[A|b], returns Ax, b, and b-Ax as vectors
    A  = getAandb(aug)[0]
    b  = getAandb(aug)[1]
    x_col_vec = vec2colVec(x)
    Ax = matMult(A,x_col_vec)
    r  = addVectors(b,scalarMult(-1.0,colVec2vec(Ax)))
    L  = [Ax,b,r]
    return(L)

def residualError(b, Ax):
    print('Residual Error Vector')
    r = []
    r = subVect(b,Ax)
    show(r)
    
def gaussElim( Ax, n):
    print("Gaussian Elimitation\n")	
    Mat = zero(n , n + 1 )# creating a new matrix		
    Mat = copyMatrix(Ax)	# pass matrix A and vector to new matrix
    	#creating a new matrix just in case Mat fails to reduce
    Cx = rowReduce( Mat )	#safe to assign new rowReduce to a new Matrix
    x = createVect( n )				#creating vector
    x = backSub( Cx )		#call backSub to get x vector
    return(x)				#return x vector

def partialPivot( A, b, n):
    d = createVect(n)   #create vector 
    d = copyVect( b )   
    Cx = augment(A,b)   #create augmented matrix
    for i in range(n):
        maxElim = abs(Cx[i][i])
        p = i
        for j in range(i+1,n):
            if abs(Cx[j][i]) > maxElim:
                maxElim = abs(Cx[j][i])
                p = j
        Cx = swaprows(Cx, p , i)
    show(Cx)
    d = gaussElim(Cx,n)
    return(d)
def scalePivot( A, b, n):
    C = copyMatrix(A)
    s = createVect(n)
    p = createVect(n)
    for i in range(n):
        sMax = 0.0
        for j in range(n):
            sMax = max(sMax, abs(C[i][j]))
            p[i] = C[i][i]
        s[i] = sMax
    for k in range(n-1):    
        rowMax = 0.0
        for i in range( k, n):
            row = abs(C[i][k] / s[i])
            if row > rowMax:
                rowMax = row
                j = i
        C = swaprows(C, j, k)
    Cx = augment( C, p)
    d = createVect(n)
    d = gaussElim(Cx,n)
    return(d)
#creating the Identity matrix I
def identityMat(n):
    I = zero( n, n)     #create 0's matrix
    for i in range(n):
        for j in range(n):
            if i == j:  #initialize pivots to 1's
                I[i][j] = 1
    return I            #return Identity matrix
#LU factorization without pivots 
def LUfactorWO( A, n):
    L = identityMat( A, n)  #initialize L with Identity matrix
    U = copyMatrix(A)       #copy A to U matrix
    for k in range(n-1):    #k .. n-1
        for j in range(k+1,n):#k+1 ..n
            L[j,i] = U[j][i] / U[k][k]  #L(j,k) = U(j,k) / U(k,k)
        for j in range(k+1,n):#k+1 .. m
            s1 = sum( L[j][k]*U[k][i] for i in range(j)) #L(j,k->n)*U(k,k->n)
            U[j][k] = U[j][k] - s1  #u(j,k->n) = U(j,k->n) - s1
    return( U, L)       #return U & L
# LU decomposition of A with pivots
def LUfactorPivot( A, n):
    U = copyMatrix(A)   
    L = identityMat(n)
    P = identityMat(n)
    PA = multMat( P, A)
    for j in range(n-1): #k ...n-1
        for i in range(j+1,n): #k+1...n
            s1 = sum(L[i][j] * U[k][j] for k in range(i)) #L(j,k->n)*U(k,k->n)
            U[i][j] = PA[i][j] - s1 #u(j,k->n) = U(j,k->n) - s1
        for i in range(j, n): #k+1 .. n
                s2 = sum(L[i][k] * U[k][j] for k in range(i)) #L(j,k->n)*U(k,k->n)
                L[i][j] = (PA[i][k] - s2) / U[j][j] #L(j,k) = U(j,k)/U(k,k) 
    return( P, L, U)
                   
def gaussSeidel( A, b, n, x):
    xNew = createVect(n)
    y = createVect(n)
    Cx = augment( A, b)
    result = ge_1(Cx)
    
    for i in range(n): #k...n
        var = 0.0
        for j in range(i):  #k..n
            var += A[i][j] * y[0][j] # Ay 
        for j in range(i+1, n):
            var += A[i][j] * x[0][j]
        y[0][i] = (b[0][i] - var)/A[i][i]
    return y
def convergTest( A, b, n, x):
    xTol = 10**(-7)
    converg = False
    while not converg:
        xNew = gaussSeidel( A, b, n, x)
        converg = True #default 
        for i in range(n):
            if abs(2*(xNew[0][i] - x[0][i])/(xNew[0][i] + x[0][i])) >= xTol:
                converge = False
                break
    return xNew
if __name__ == '__main__':
    A = [[10,10,10,10**17],
         [1,10**-3,10**-3,10**-3],
         [20,1,10**-3,10**-3],
         [1,1,1,10**-3]]
    n = rows(A)
    m = cols(A)
    show(A)
    b = [10**17, 1,2,3]
    x = createVect(n)
    Ax = zero(n, m+1)
    Ax = augment(A,b)
    x = gaussElim(Ax, n)
    print('Gaussian Elim')
    show(x)
    r = subVectors(b,x)
    show(r)
    print('Error of X')
    xb = createVect( n )
    print('Partial Pivot')
    xb = partialPivot( A, b, n)
    show(xb)
    xc = scalePivot(A, b, n)
    print('Scale Pivot')
    show(xc)
