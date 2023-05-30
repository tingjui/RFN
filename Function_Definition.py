import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv
import copy


class kernel_logistic_regression(object):
    def __init__(self, lamb = 0.0,w0=None,w_opt=None,l_opt=None,kappa=None, hatkappa=None):
        self.lamb = lamb
        
    def loss_computing(self, func_value, real_value):
        loss = np.sum(np.log(1+np.exp(-np.multiply(func_value, real_value))))
        loss = loss/real_value.shape[0]

        return loss

    def loss_regularization(self, w, K):
        r = np.dot(np.dot(w, K), w)
        
        return self.lamb*r
    
    def hessian_middle_flat(self, func_value, real_value):   
        d = np.zeros(func_value.shape[0])
        d = np.exp(-np.multiply(func_value, real_value))/((1+np.exp(-np.multiply(func_value, real_value)))**2)
        d = d/real_value.shape[0]

        return d
    
    def gradient_computing(self, func_value, real_value, K):
        g = -real_value*np.exp(-np.multiply(func_value, real_value))/(1+np.exp(-np.multiply(func_value, real_value)))
        g = np.dot(K, g)
        g = g/real_value.shape[0]

        return g
    
def gram_matrix_calculation(data, kernel, squared_sigma):
    """
    Description
    -----------
    Calculate the gram matrix of the training data matrix based on the given kernel function.
    -----------
    
    Input
    -----------
    data:               the matrix of the training data
    kernel:             the type of the kernel funciton
    squared_sigma       the bandwidth of the kernel function (for Gaussian kernel)
    -----------
    
    Output
    -----------
    gram_matrix         the corresponding gram matrix
    -----------
    """
    
    gram_matrix = np.zeros((data.shape[0], data.shape[0]))
    
    if kernel == 'Gaussian':
        for i in range(data.shape[0]):
            gram_matrix[i, :] = np.sum((data[i,:]-data)**2, axis=1)
        
        gram_matrix = np.exp(-squared_sigma*gram_matrix/(2))
        
                
    return gram_matrix   

def gram_matrix_approximating(data, kernel, num_rf, squared_sigma):
    """
    Description
    -----------
    Approximate the original gram matrix with random features. (Approximate K as QQ^T)
    -----------
    
    Input
    -----------
    data:               the matrix of the training data
    kernel:             the type of the kernel funciton
    num_rf:             the total number of random features
    squared_sigma       the bandwidth of the kernel function (for Gaussian kernel)
    -----------
    
    Output
    -----------
    Q                   the low-rank approximation of the original gram matrix
    -----------
    """
    
    if kernel == 'Gaussian':
        b = np.random.uniform(0, 2*np.pi, num_rf)
        w = np.random.multivariate_normal(np.zeros(data.shape[1]), squared_sigma*np.eye(data.shape[1]), num_rf)
        Q = np.sqrt(2.0/num_rf)*np.cos(np.dot(data, w.T)+b)
        
    return (Q, w, b)
    
def back_tracking(problem, w, direct, gradient, K, y, initial_step_size=1, alpha=0.3, beta=0.8, normalized_step=False):
    """
    Description
    -----------
    Find the suitable step size t that satisfies the following inequality by using Armijo line search:
    L(w + t*d) <= L(w) + t*alpha*L'(w)^Td, where 
    L:          the loss function
    w:          the current iterate
    t:          the step size
    alpha:      a constant in (0, 1)
    d:          the Newton direction
    -----------
    
    Input
    -----------
    problem:
        problem.lamb                    the weight of the regularization term
        problem.loss_computing          function to compute the loss of kernel logistic regression
        problem.loss_regularization     function to compute the loss of the regularization term
        problem.hessian_middle_flat     function to compute the diagonal D
        problem.gradient_computing      function to compute the gradient
        
    w:                                  the current iterate
    direct:                             the Newton direction
    gradient:                           the current gradient
    K:                                  the original gram matrix
    y:                                  the matrix of the training label
    initial_step_size:                  the initial step size
    alpha:                              hyper parameter of backtracking line search (in (0, 1))
    beta:                               hyper parameter of backtracking line search (in (0, 1))
    normalized_step:                    whether to preprocess the initial step size
    -----------
    
    Output
    -----------
    t:                                  the step size satisfying the Armijo–Goldstein condition 
    -----------
    """
    
    backtrack_count = 0
    
    multi = 50
    if normalized_step==True:
        t = 1*multi/(np.absolute(np.dot(gradient,direct)))
        if t > 1:
            t=1
    else:
        t=initial_step_size
    
    l = problem.loss_computing(np.dot(K, w), y) + problem.loss_regularization(w,K)
    m = alpha*np.dot(gradient,direct)
    while(problem.loss_computing(np.dot(K, w+t*direct), y) + problem.loss_regularization(w+t*direct,K) >\
          l+ t*m):
        t = t*beta
        backtrack_count = backtrack_count + 1

        

    return (t, backtrack_count)
    
    
def lbfgs(g, s, y, rho, Hdiag=None):
    (p, k) = s.shape
    
    
    r = np.zeros((p,k+1))
    al =np.zeros(k)
    be =np.zeros(k)

    q = g

    for i in range(k):
        al[k-1-i] = rho[k-1-i]*s[:,k-1-i].dot(q)
        q = q-al[k-1-i]*y[:,k-1-i]
    

    # Multiply by Initial Hessian
    Hdiag = (s[:,k-1].dot(y[:,k-1]))/(y[:,k-1].dot(y[:,k-1]))
    r[:,0] = Hdiag*q

    for i in range(k):
        be[i] = rho[i]*y[:,i].dot(r[:,i])
        r[:,i+1] = r[:,i] + s[:,i]*(al[i]-be[i])
    
    return r[:,k]

def conjugate_grad(A, b, mu, low_rank=False, low_rank_A1=None, low_rank_A2=None, low_rank_B1=None, low_rank_B2=None, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    -----------
    
    
    Input
    -----------
    A:                 a positive semi-definite matrix
    b:                 a vector
    x:                 the vector of initial point
    mu:                the coefficient of the nugget term
    low_rank:          whether to approximate the matrix A as a low-rank factorization plus a nugget term: A ~= A1A2 + B1B2 + mu*I
    low_rank_A1:       A1
    low_rank_A2:       A2
    low_rank_B1:       B1
    low_rank_B2:       B2
    -----------
    
    
    Returns
    -----------
    x:                 the solution of Ax = b
    -----------
    """
    n = len(b)
    norm_b = np.linalg.norm(b, 2)
    if not x:
        x = np.zeros(n)
        
    if not low_rank: 
        r = np.dot(A, x) - b
        p = - r
        r_k_norm = np.dot(r, r)
        for i in range(n):
            Ap = np.dot(A, p)
            alpha = r_k_norm / np.dot(p, Ap)
            x += alpha * p
            r += alpha * Ap
            r_kplus1_norm = np.dot(r, r)
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm
            if np.linalg.norm(r, 2)/norm_b < 1e-6:
                print('Itr:', i)
                break
            p = beta * p - r
    else:
        r = np.dot(low_rank_A1, np.dot(low_rank_A2, x)) + np.dot(low_rank_B1, np.dot(low_rank_B2, x)) + mu*x - b
        p = - r
        r_k_norm = np.dot(r, r)
        for i in range(n):
            Ap = np.dot(low_rank_A1, np.dot(low_rank_A2, p)) + np.dot(low_rank_B1, np.dot(low_rank_B2, p))+mu*p
            alpha = r_k_norm / np.dot(p, Ap)
            x += alpha * p
            r += alpha * Ap
            r_kplus1_norm = np.dot(r, r)
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm
            if np.linalg.norm(r, 2)/norm_b < 1e-6:
                print('Itr:', i)
                break
            p = beta * p - r
    return x

def Newton_solver_rf(RF_matrix, train_label, problem, itera, initial_w):
    w = initial_w
    
    t = np.zeros(itera+1)   
    l = np.zeros(itera+1)
    sols = []
    results = {}
    
    t[0] = 0
    

    l[0] = problem.loss_computing(np.dot(RF_matrix,w), train_label) + problem.lamb*np.dot(w,w)
    
    sols.append(copy.deepcopy(w))
    
    for i in range(itera):
        loss = problem.loss_computing(np.dot(RF_matrix,w), train_label) + problem.lamb*np.dot(w,w)
        print(loss)
        start_time = time.time()
    
    
        D = problem.hessian_middle_flat(np.dot(RF_matrix,w), train_label)
        h = np.multiply(RF_matrix.T, D).dot(RF_matrix) + 2*problem.lamb*np.eye(w.shape[0])
        g = problem.gradient_computing(np.dot(RF_matrix,w), train_label, RF_matrix.T) + 2*problem.lamb*w
        
        print("Gradient norm", np.linalg.norm(g))

        direct = - np.linalg.solve(h, g)

        tt = 1
        #tt = back_tracking(problem, w, direct, g, K, train_label,alpha=0.3, beta=0.5)

        w = w + tt*direct
        
        time_iter = (time.time() - start_time)        
        t[i+1] = time_iter
        
        sols.append(copy.deepcopy(w))
        l[i+1] = problem.loss_computing(np.dot(RF_matrix,w), train_label) + problem.lamb*np.dot(w,w)
        

    print("Newton solver (RF) ends")

    print("Further postprocessing ......")

    for i in range(len(t)-1):
        t[i+1] += t[i]
    
    results['t'] = t
    results['l'] = l
    
   
    print("Done! :)")
    
    return (sols,results)
    
def GD_solver(K, train_label, problem, itera, initial_w):
    w = initial_w
    
    t = np.zeros(itera+1)   
    l = np.zeros(itera+1)
    sols = []
    results = {}
    
    t[0] = 0
    

    l[0] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
    
    sols.append(copy.deepcopy(w))
    
    for i in range(itera):
        loss = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        print(loss)
        start_time = time.time()
    
    
#         D = problem.hessian_middle_flat(np.dot(K,w), train_label)
#         h = np.multiply(K, D).dot(K) + 2*problem.lamb*K
        g = problem.gradient_computing(np.dot(K,w), train_label, K) + 2*problem.lamb*np.dot(K,w)
        

        direct = -g

#         tt = 1
        (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label,alpha=0.3, beta=0.3)

        w = w + tt*direct
        
        time_iter = (time.time() - start_time)        
        t[i+1] = time_iter
        
        sols.append(copy.deepcopy(w))
        l[i+1] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        
        print("step size: ",tt)
        

    print("GD solver ends")

    print("Further postprocessing ......")

    for i in range(len(t)-1):
        t[i+1] += t[i]
    
    results['t'] = t
    results['l'] = l
    
   
    print("Done! :)")
    
    return (sols,results)

def Newton_solver(K, train_label, problem, itera, initial_w):
    """
    Description
    -----------
    The Newton step is computed exactly with the original Hessian and the full gradient.
    -----------
    
    Input
    -----------
    problem:
        problem.lamb                    the weight of the regularization term
        problem.loss_computing          function to compute the loss of kernel logistic regression
        problem.loss_regularization     function to compute the loss of the regularization term
        problem.hessian_middle_flat     function to compute the diagonal D
        problem.gradient_computing      function to compute the gradient
        
    K:                                  the original gram matrix
    train_label:                        the matrix of the training labels
    itera:                              the number of total optimization iterations
    initial_w:                          the initial iterate
    -----------
    
    Output
    -----------
    sols:                               solution of each iteration
    results:
        results['t']                    running time of each iteration
        results['l']                    loss value of each iteration
    -----------
    """
    
    w = initial_w
    
    t = np.zeros(itera+1)   
    l = np.zeros(itera+1)
    sols = []
    results = {}
    
    t[0] = 0
    

    l[0] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
    
    sols.append(copy.deepcopy(w))
    
    for i in range(itera):
        loss = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        print(loss)
        start_time = time.time()
    
    
        D = problem.hessian_middle_flat(np.dot(K,w), train_label)
        h = np.multiply(K, D).dot(K) + 2*problem.lamb*K
        g = problem.gradient_computing(np.dot(K,w), train_label, K) + 2*problem.lamb*np.dot(K,w)
        

        direct = - np.linalg.solve(h, g)

#         tt = 1
        (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label,alpha=0.3, beta=0.3)

        w = w + tt*direct
        
        time_iter = (time.time() - start_time)        
        t[i+1] = time_iter
        
        sols.append(copy.deepcopy(w))
        l[i+1] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        
        print("step size: ",tt)
#         print("Condition Number: ", np.linalg.cond(h))
#         print("Condition Number (D): ", np.max(D)/np.min(D))
        

    print("Newton solver ends")

    print("Further postprocessing ......")

    for i in range(len(t)-1):
        t[i+1] += t[i]
    
    results['t'] = t
    results['l'] = l
    
   
    print("Done! :)")
    
    return (sols,results)

def LBFGS_solver(K, train_label, problem, bfgs_m, itera, initial_w):    
    """
    Description
    -----------
    The Newton step is computed inexactly by BFGS algorithm.
    -----------
    
    Input
    -----------
    problem:
        problem.lamb                    the weight of the regularization term
        problem.loss_computing          function to compute the loss of kernel logistic regression
        problem.loss_regularization     function to compute the loss of the regularization term
        problem.hessian_middle_flat     function to compute the diagonal D
        problem.gradient_computing      function to compute the gradient
        
    K:                                  the original gram matrix
    train_label:                        the matrix of the training labels
    bfgs_m:                             the memory size for BFGS algorithm
    itera:                              the number of total optimization iterations
    initial_w:                          the initial iterate
    -----------
    
    Output
    -----------
    sols:                               solution of each iteration
    results:
        results['t']                    running time of each iteration
        results['l']                    loss value of each iteration
    -----------
    """
    
    w = initial_w
    
    
    t = np.zeros(itera+1)
    sols = []
    l = np.zeros(itera+1)
    results = {}
    
    g_prev = 0
    y = np.zeros((train_label.shape[0],0))
    S = np.zeros((train_label.shape[0],0))
    rho = np.zeros(0)
    
    t[0] = 0
    sols.append(initial_w)
    l[0] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
    
    for i in range(itera):
        loss = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        print(loss)
        
        
        
        start_time = time.time()
            
        func_value = np.dot(K,w)
        g = problem.gradient_computing(func_value, train_label, K) + 2*problem.lamb*func_value
        
        if i == 0:
            direct = -g
            
        else:
            s_prev = w - sols[i-1]
            y_prev = g - g_prev;
                     
            rho_prev = np.expand_dims(1/(s_prev.dot(y_prev)), axis=0)
       

            if S.shape[1] >= bfgs_m:
                S = np.concatenate((S[:, 1:S.shape[1]], np.expand_dims(s_prev, axis=1)), axis=1)
                y = np.concatenate((y[:, 1:y.shape[1]], np.expand_dims(y_prev, axis=1)), axis=1)
                rho = np.concatenate((rho[1:y.shape[1]], rho_prev), axis=0)
            else:
                S = np.concatenate((S, np.expand_dims(s_prev, axis=1)), axis=1)
                y = np.concatenate((y, np.expand_dims(y_prev, axis=1)), axis=1)
                rho = np.concatenate((rho, rho_prev), axis=0)
                                
            direct = -lbfgs(g, S, y, rho)

        g_prev = g;
        
        

        (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label, alpha=0.3, beta=0.3)
        
        
        w = w + tt*direct
        
        time_iter = (time.time() - start_time)  
        t[i+1] = time_iter      
        sols.append(copy.deepcopy(w))
        l[i+1] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        
        print("step size: ",tt)
        
    print("L-BFGS solver ends")

    print("Further postprocessing ......")

    for i in range(len(t)-1):
        t[i+1] += t[i]
    
    results['t'] = t
    results['l'] = l

    print("Done! :)")
    
    return (sols,results)

def SSN_uniform_CG(K, train_label, num_sub_newton, problem, mu, itera, initial_w):
    """
    Description
    -----------
    The Newton step is computed inexactly by CG algorithm with the full gradient and the approximated Hessian generated from SSNCG.
    -----------
    
    Input
    -----------
    problem:
        problem.lamb                    the weight of the regularization term
        problem.loss_computing          function to compute the loss of kernel logistic regression
        problem.loss_regularization     function to compute the loss of the regularization term
        problem.hessian_middle_flat     function to compute the diagonal D
        problem.gradient_computing      function to compute the gradient
        
    K:                                  the original gram matrix
    train_label:                        the matrix of the training labels
    num_sub_newton:                     the size of the sub-sampled data set
    mu:                                 the coefficient of the nugget term
    itera:                              the number of total optimization iterations
    initial_w:                          the initial iterate
    -----------
    
    Output
    -----------
    sols:                               solution of each iteration
    results:
        results['t']                    running time of each iteration
        results['l']                    loss value of each iteration
    -----------
    """
    w = initial_w

    
    t = np.zeros(itera+1)
    sols = []
    l = np.zeros(itera+1)
    results = {}
    
    step_size_prev = 0
    
    t[0] = 0
    sols.append(initial_w)
    l[0] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)

    for i in range(itera):
        loss = problem.loss_computing(np.dot(K,w), train_label,) + problem.loss_regularization(w, K)
        print(loss)


        start_time = time.time()


        idx = np.random.choice(int(train_label.shape[0]),int(num_sub_newton))

        KVI = K[:, idx]
        KIV = K[idx, :]
        K_sub = KIV
        KII = K[:, idx][idx, :] 

        KII_pinv = np.linalg.pinv(KII)

        D = problem.hessian_middle_flat(np.dot(KIV, w), train_label[idx])
 
        g = problem.gradient_computing(np.dot(K,w), train_label, K) + 2*problem.lamb*np.dot(K,w)


        direct = -conjugate_grad(K, g, mu, True, np.multiply(KVI, D), KIV, 2*problem.lamb*np.dot(KVI, KII_pinv), KIV)


        if i==0:
            (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label, alpha=0.3, beta=0.1,\
                               normalized_step=True)
        else:
            (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label, 2*step_size_prev, alpha=0.3, beta=0.1, normalized_step=True)
            
        w = w + tt*direct
        
        time_iter = (time.time() - start_time)
        t[i+1] = time_iter
        step_size_prev = tt

        sols.append(copy.deepcopy(w))

        l[i+1] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        
    print("SSN solver (CG) ends")

    print("Further postprocessing ......")

    for i in range(len(t)-1):
        t[i+1] += t[i]
    
    results['t'] = t
    results['l'] = l

    
    print("Done! :)")
    
    return (sols,results)

def RFN_inversion_lemma(K, train_data, train_label, num_rf, problem, mu, squared_sigma, itera, initial_w):    
    """
    Description
    -----------
    The Newton step is computed exactly by the matrix inversion lemma with the full gradient and the approximated Hessian generated from         RFN.
    -----------
    
    Input
    -----------
    problem:
        problem.lamb                    the weight of the regularization term
        problem.loss_computing          function to compute the loss of kernel logistic regression
        problem.loss_regularization     function to compute the loss of the regularization term
        problem.hessian_middle_flat     function to compute the diagonal D
        problem.gradient_computing      function to compute the gradient
        
    K:                                  the original gram matrix
    train_data:                         the matrix of the training data
    train_label:                        the matrix of the training labels
    num_rf:                             the number of all random features used for Hessian approximation
    mu:                                 the coefficient of the nugget term
    squared_sigma:                      the bandwidth of the Gaussian kernel funciton
    itera:                              the number of total optimization iterations
    initial_w:                          the initial iterate
    -----------
    
    Output
    -----------
    sols:                               solution of each iteration
    results:
        results['t']                    running time of each iteration
        results['l']                    loss value of each iteration
    -----------
    """
    w = initial_w
    num_rf = int(num_rf)
    
    t = np.zeros(itera+1)
    sols = []
    l = np.zeros(itera+1)
    results = {}
    
    t[0] = 0
    sols.append(initial_w)
    l[0] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
    
    step_size_prev = 0

    for i in range(itera):
        loss = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)
        print(loss)

        start_time = time.time()
        Q, w_rf, b_rf = gram_matrix_approximating(train_data, 'Gaussian', num_rf, squared_sigma)
        
        
        func_value = np.dot(K,w)        
        g = problem.gradient_computing(func_value, train_label, K) + 2*problem.lamb*func_value
        
        
        c = np.multiply(Q.T, problem.hessian_middle_flat(func_value, train_label)).dot(Q) + 2*problem.lamb*np.eye(num_rf)   
        
        
        middle = inv(c)*mu + Q.T.dot(Q)
        
        direct = -(g-Q.dot(np.linalg.solve(middle, Q.T.dot(g))))/mu
        
        
        if i==0:
            (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label, alpha=0.3, beta=0.1,\
                               normalized_step=True)
        else:
            (tt, backtrack_count) = back_tracking(problem, w, direct, g, K, train_label, 2*step_size_prev, alpha=0.3, beta=0.1, normalized_step=True)
        
        w = w + tt*direct
        
        time_iter = (time.time() - start_time)
        
        t[i+1] = time_iter       
        sols.append(copy.deepcopy(w))
        l[i+1] = problem.loss_computing(np.dot(K,w), train_label) + problem.loss_regularization(w, K)

        step_size_prev = tt
        
        print("Back tracking iteration: ", backtrack_count)
    
        
    print("RFN solver (inversion lemma) ends")

    print("Further postprocessing ......")
    

    
    for i in range(len(t)-1):
        t[i+1] += t[i]
    
    results['t'] = t
    results['l'] = l
    
    print("Done! :) haha\haha")
    
    return (sols,results)