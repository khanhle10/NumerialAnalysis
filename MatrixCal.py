
def transpose(mat):
#    "return transpose of mat"
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
    return([[M[row][col] for col in range(cols(M))]for row in
            range(rows(M))])

def addrows(M, f, t, scale=1):
#    "add scale times row f to row t"
    N=copyMatrix(M)
    T=addVectors(scalarMult(scale,N[f]),N[t])
    N[t]=T
    return(N)
def zero(m,n):
#    "Create zero matrix"
    new_mat = [[0 for col in range(n)] for row in range(m)]
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
def createVect():
  b = [0 for i in range(n)]
  for i in range(0, rows(b)):
      while True:
       element = input("Enter NUmber: ")
       if element > 0:
          b[i] = element
          break
  return(b)
def show(mat):
#    "Print out matrix"
    for row in mat:
        print(row)
        
def findPivotrow1(mat,col):
#    Finds index of the first row with nonzero entry on or
#    below diagonal.  If there isn't one return(-1).

    epsilon = 10**(-17)
    for row in range(col, rows(mat)):
#        if mat[row][col] != 0:
        if abs(mat[row][col]) > epsilon:
            return(row)
    return(-1)

def swaprows(M,i,j):
#    "swap rows i and j in matrix M"
    N=copyMatrix(M)
    T = N[i]
    N[i] = N[j]
    N[j] = T
    return N

def subVectors(S,T):
    "return S+T"
    C = []
    for i in range(len(S)):
        for j in range(len(T[0])):
          C[i][j] = S[i][j] - T[i][j]
    return(C)

def multMat(mat1,mat2):
    "multiply two matrices"
    result = [[]]
    result = [[sum(i*j for i,j in zip(mat1Row, mat2Col)) for mat2Col in zip(*mat2)] for mat1Row in mat1]
    return(result)

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
    return(N)


def backSub(M):

#   given a row reduced augmented matrix with nonzero 
#   diagonal entries, returns a solution vector
    

    cs = cols(M)-1 # cols not counting augmented col
    sol = [0 for i in range(cs)] # place for solution vector
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
def gaussElimA(A,b, n, m):
    print("Gaussian Elimitation\n")
    Mat = zero(n,m+1)
    show(Mat)
    Mat = augment(A,b)
    show(Mat)
    newMat = copyMatrix(Mat)
    newMat = rowReduce(Mat)
    show(newMat)
    x = []
    x = backSub(Mat)
    show(x)
    return(x)

def LUfactor(A, b, n , m ):
    print("Gaussian Elimitation\n")
    
    C = zero(n, m+1)
    C = copyMatrix(A)
    p = range(n)
    scal = [0]*n
    for i in xrange(n):
        sMax = 0.0
        for j in xrange(n):
            sMax = max(sMax, abs(C[i,j]))
        scale[i] = sMax
    for k in xrange(n - 1):
        rowMax = 0.0
        for i in xrange(k,n):
            row = abs(C[p[i],k] / scale[p[i]])
            if row > rowMax:
                rowMax = row;
                j = i;
        newC = copyMatrix(C)
        newC = swaprows(C, k, j)
        newC = rowReduce(newC)
    return(newC, p)

def scalePartPivot( A, b, n, m):
    
    C = zero(n, m+1)
    C = augment(A,b)
    for i in range(n):
        for col in 
def matrixInputAandb():
    while True:
      n = int (input("Please Enter Row Size: "))
      if n > 0:
        m = int (input("Please Enter Col Size: "))
        if m > 0:
         A = zero(n, m)
         break
    return(A)
if __name__ == '__main__':
    A = [[10,10,10,10**17],
         [1,10**-3,10**-3,10**-3],
         [1,1,10**-3,10**-3],
         [1,1,1,10**-3]]
    n = rows(A)
    m = cols(A)
    show(A)
    b = [10**17, 1,2,3]
    show(b)
    gaussElimA(A,b, n,m)
    x = gaussElimB(A,b,n,m)
    show(x)
    r = subVectors(b,x)
    xb = []
    xb = partialPivot(A,b,n,m)
    
    








	    
	    
