###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

def Beta_Gradient_Descent_Matrix_Factorization(Observed_Matrix, Users_Matrix, Items_Matrix, Features_Number, steps=5000, learning_rate=0.0002, beta=0.02):
    
    Items_Matrix = Items_Matrix.T
    
    for step in range(steps):
        
        for row in range(len(Observed_Matrix)):
            for col in range(len(Observed_Matrix[row])):
                if Observed_Matrix[row][col] > 0:
                    prediction = numpy.dot(Users_Matrix[row,:],Items_Matrix[:,col])
                    target = Observed_Matrix[row][col]
                    local_error = target - prediction
                    for feat in range(Features_Number):
                        Users_Matrix[row][feat] = Users_Matrix[row][feat] + learning_rate * (2 * local_error * Items_Matrix[feat][col] - beta * Users_Matrix[row][feat])
                        Items_Matrix[feat][col] = Items_Matrix[feat][col] + learning_rate * (2 * local_error * Users_Matrix[row][feat] - beta * Items_Matrix[feat][col])
                        
                        
        eR = numpy.dot(Users_Matrix, Items_Matrix)
        e = 0
        for row in range(len(Observed_Matrix)):
            for col in range(len(Observed_Matrix[row])):
                if Observed_Matrix[row][col] > 0:
                    e = e + pow(Observed_Matrix[row][col] - numpy.dot(Users_Matrix[row,:],Items_Matrix[:,col]), 2)
                    for feat in range(Features_Number):
                        e = e + (beta/2) * ( pow(Users_Matrix[row][feat],2) + pow(Items_Matrix[feat][col],2) )
        if e < 0.001:
            break
    return Users_Matrix, Items_Matrix.T

###############################################################################

if __name__ == "__main__":
    Observed_Matrix = numpy.array([
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ])

    N = len(Observed_Matrix)
    M = len(Observed_Matrix[0])
    Features_Number = 2

    Users_Matrix = numpy.random.rand(N,Features_Number)
    Items_Matrix = numpy.random.rand(M,Features_Number)

    Predicted_Users_Matrix, Predicted_Items_Matrix = Beta_Gradient_Descent_Matrix_Factorization(Observed_Matrix, Users_Matrix, Items_Matrix, Features_Number)
    print(numpy.dot(Predicted_Users_Matrix, Predicted_Items_Matrix.T))
