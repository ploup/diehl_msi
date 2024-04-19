

# coding: utf-8

# In[1]:

import numpy as np

from scipy.special import gammaln

def psi(x):

    r = 0;

    while (x<=5):
        r -= 1/x
        x += 1

    f = 1/(x*x);

    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
                   + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))));

    return r + np.log(x) - 0.5/x + t;

#La log-vraisemblance négative

def neg_LL(c,alpha_beta, rpt, pro, trials): ## c = 1/phi -1  ou phi = 1/(c+1)
    n_samples, _ = rpt.shape
    alpha = alpha_beta[0]
    beta = alpha_beta[1:]
    y = trials*pro
    pi =  np.exp(alpha)/(1+np.exp(-beta[0]-beta[1]*rpt))

    LL_array = 0
    for k in np.arange(n_samples) :
        LL_k= - gammaln(c+trials[k])+ gammaln(c) + gammaln(c * pi[k] + y[k]) - gammaln(c * pi[k] ) +gammaln(c * (1 - pi[k]) + trials[k] -y[k]) - gammaln( c * (1- pi[k])) 
        LL_array += LL_k
    return  - LL_array / n_samples
    



# Le gradient de la log-vraisemblance négative

def neg_LL_grad_k(c,alpha_beta, k,rpt_k, pro_k, trials_k):
    alpha = alpha_beta[0]
    beta = alpha_beta[1:]
    y_k = trials_k*pro_k
    features_k_dot_beta = beta[0]+beta[1]*rpt_k+beta[k+2]
    p = beta.shape[0]
    w_beta_k = np.zeros(alpha_beta.shape[0]-1)

    pi =  np.exp(alpha)/(1+np.exp(-features_k_dot_beta))#np.exp(phi + features.dot(beta.T))/(1+np.exp(features.dot(beta.T)))



    Tnk = - psi(c+trials_k)+ psi(c) 
    Tyk = psi(c * pi + y_k) - psi(c * pi )   
    Tnk_yk = psi(c * (1 - pi) + trials_k - y_k) - psi( c * (1- pi)) 

    w_c = - Tnk + pi *Tyk +  (1 - pi)*Tnk_yk
    grad_pi_k =  c * Tyk - c * Tnk_yk
    w_alpha = pi* grad_pi_k
   
    #print(grad_pi_k,Tnk_yk,Tyk)
    w_beta_k[0]= pi*(1+np.exp(features_k_dot_beta))**(-1)*grad_pi_k
  
    w_beta_k[1]=rpt_k * pi*(1+np.exp(features_k_dot_beta))**(-1)* grad_pi_k
        #print((rpt[k]))# / v[k])* grad_pi_k)
    w_beta_k[k+2]=pi*(1+np.exp(features_k_dot_beta))**(-1)*grad_pi_k
    w_beta_b = np.asarray(w_beta_k).reshape((p,))
#    print(np.asarray(w_beta).reshape((p,)))
    minus_LL_grad =  -np.hstack([ w_c,w_alpha,w_beta_b])
    return minus_LL_grad


# SGD

def sgd(c_alpha_beta0,  rpt ,pro, trials, l_l1 = 0., n_iter=100, learning_rate =1.):
    """Stochastic gradient descent algorithm."""
    
    n_samples,_ = rpt.shape
    
    c_alpha_beta = c_alpha_beta0.copy(); c_alpha_beta_new = c_alpha_beta0.copy() 
    #features = features.tocsr()
    

    objectives = []

    
    def f(c_alpha_beta, rpt, pro, trials):
        c = c_alpha_beta[0]
        alpha_beta = c_alpha_beta[1:]
        return neg_LL(c,alpha_beta, rpt, pro, trials)
    
    def grad_f_k(c_alpha_beta,k, rpt_k, pro_k, trials_k):
        c = c_alpha_beta[0]
        alpha_beta = c_alpha_beta[1:]
        return neg_LL_grad_k(c,alpha_beta, k,rpt_k, pro_k, trials_k)

    def g(c_alpha_beta,l_l1):
        return l_l1*np.sum(np.abs(c_alpha_beta[4:]))
    
    def prox_g(c_alpha_beta,l_l1,step):
        prox_l1 = np.sign(c_alpha_beta[4:]) * (np.abs(c_alpha_beta[4:]) - l_l1 * step ) * (np.abs(c_alpha_beta[4:]) > l_l1 *step )
        c_alpha_beta[4:] =  prox_l1
        return c_alpha_beta 


    obj = f(c_alpha_beta, rpt, pro, trials)+g(c_alpha_beta,l_l1)
    objectives.append(obj)
#    objectives_cum.append(obj)
    
#    x_cum = x

    print("")
    print("Lauching SGD solver...")
    print(' | '.join([name.center(8) for name in ["it", "obj"]]))
    for k in range(n_iter):
        for idx in range(n_samples):
            # generate random indice
            i =  np.random.randint(0, n_samples, 1)[0]
            pro_i = pro[i]
            trials_i = trials[i]
            #features_i = features[i,:].toarray()[0]
            step = learning_rate / np.sqrt((k*n_samples+ idx+ 1))
            c_alpha_beta_new[:] = c_alpha_beta - step * grad_f_k(c_alpha_beta,i, rpt[i], pro_i, trials_i)
            c_alpha_beta_new[:] = prox_g(c_alpha_beta_new,l_l1,step)           
            #c_alpha_beta_new[1] = c_alpha_beta_new[1]#0#np.min(np.max(-0.1,c_alpha_beta_new[1]),0.)
            c_alpha_beta = c_alpha_beta_new#((k*n_samples+ idx )*c_alpha_beta + c_alpha_beta_new) / (k*n_samples+ idx+ 1) 
            #if idx % 10000 == 0:   
            #    print(idx)

        obj = f(c_alpha_beta, rpt, pro, trials)+g(c_alpha_beta,l_l1)
        #obj_cum = f(x_cum/((k+1)*idx))+g(x_cum/((k+1)*idx))
        #err = norm(x - x_min) / norm(x_min)
        #err_cum = norm(x_cum/((k+1)*idx) - x_min) / norm(x_min)
        #errors.append(err)
        #errors_cum.append(err)
        #if obj > objectives[-1]:
        #    print("problème de pas")
        #    break
            
        objectives.append(obj)
        
        if k % 20 == 0:
           
           # print x_cum
            print(' | '.join( [ ("%d" % k).rjust(8), ("%.4e" % obj).rjust(8)])) #, ("%.2e" % err).rjust(8)]))
           
    return c_alpha_beta, objectives#, objectives_cum ,errors_cum