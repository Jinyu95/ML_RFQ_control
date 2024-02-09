import numpy as np


def objectives(U_list, trainu_con, trainx_con, model, sc_x, sc_d, sc_i, target):
    # U_list: multiple valve setting combinations
    # trainu_con: include control inputs, 
    # [0, 1, 2, 3, 4, 5, 9, 10, 11, 12] are the delayed variables, 
    # [6, 7, 8] are the instant variables
    delay_data = np.zeros((len(U_list), 120, 10))
    for i in range(len(U_list)):
        u1 = U_list[i][0]
        u2 = U_list[i][1]
        
        delay1 = trainu_con[0:120,[0, 1, 2, 3, 4, 5, 9, 10, 11, 12]] 
        delay1[60:,0:8] = delay1[59,0:8] # untunable variables, unknown for the last 60 points
        delay1[60:,8] = u1 # CV2 setting
        delay1[60:,9] = u2 # CV3 setting
        delay1 = sc_d.transform(delay1).reshape((1,120,10))
        delay_data[i,:,:] = delay1[0,:,:]

    instant1 = trainu_con[59, [6, 7, 8]] # instant variables 
    frq1 = trainx_con[:60,:] # recorded frequency 
    
    instant1 = sc_i.transform(instant1.reshape((1,3)))
    frq1 = sc_x.transform(frq1.reshape((60,1))).reshape((1,60))
    
    instant_data = np.tile(instant1, (len(U_list), 1))
    frq_data = np.tile(frq1, (len(U_list), 1))

    optimized_f = model.predict([delay_data, instant_data, frq_data]) # predictions
    err_list = []
    for i in range(len(U_list)):
        optimized_f_real = sc_x.inverse_transform(optimized_f[i].reshape(60,1))
        optimized_f_real_sq = optimized_f_real**2
        # weights = np.geomspace(1, 1000, 60)
        weights = np.ones(60)
        weights = weights / np.sum(weights)
        weighted_sum = np.sum(optimized_f_real_sq.flatten() * weights)
        err_list.append(weighted_sum)
    return err_list

def getfreq(u1, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target):
    delay1 = trainu_con[0:120, [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]] # delayed variables
    delay1[60:,0:8] = delay1[59,0:8] # untunable variables, unknown for the last 60 points
    delay1[60:,8] = u1 # CV2 setting
    delay1[60:,9] = u2 # CV3 setting
    instant1 = trainu_con[59, [6, 7, 8]] # instant variables 
    frq1 = trainx_con[0:60, :]

    
    delay1 = sc_d.transform(delay1).reshape((1,120,10))
    instant1 = sc_i.transform(instant1.reshape((1,3)))
    frq1 = sc_x.transform(frq1.reshape((60,1))).reshape((1,60))

    optimized_f = model.predict([delay1, instant1, frq1]) # predictions
    optimized_f_real = sc_x.inverse_transform(optimized_f.reshape(60,1))
    optimized_f_real_sq = optimized_f_real**2
    # weights = np.geomspace(1, 1000, 60)
    weights = np.ones(60)
    weights = weights / np.sum(weights)
    weighted_sum = np.sum(optimized_f_real_sq.flatten() * weights)
    return weighted_sum, optimized_f_real

def approximate_hessian(u1, u2, trainu_con, trainx_con, model, sc_x, sc_d, sc_i, target, epsilon=1e-1):
    hessian = np.zeros((2, 2))

    # Original value of the function
    u_original = np.array([u1, u2])
    u_list = [u_original]

    for i in range(2):
        for j in range(2):
            # Perturb parameters i and j
            u1_i, u2_i = perturb(u1, u2, i, epsilon)
            u1_j, u2_j = perturb(u1, u2, j, epsilon)
            u1_ij, u2_ij = perturb(u1_i, u2_i, j, epsilon)
            u_list.append(np.array([u1_i, u2_i]))
            u_list.append(np.array([u1_j, u2_j]))
            u_list.append(np.array([u1_ij, u2_ij]))
    objs = objectives(u_list, trainu_con, trainx_con, model, sc_x, sc_d, sc_i, target)
    f_original = objs[0]
    for i in range(2):
        for j in range(2):
            # Calculate the function values after perturbation
            f_i = objs[3*(i*2+j)+1]
            f_j = objs[3*(i*2+j)+2]
            f_ij = objs[3*(i*2+j)+3]

            # Approximate the second derivative
            hessian[i, j] = (f_ij - f_j - f_i + f_original) / (epsilon ** 2)

    return hessian

def perturb(u1, u2, index, epsilon):
    if index == 0:
        u1 += epsilon
    elif index == 1:
        u2 += epsilon
    return u1, u2

def approximate_gradient(u1, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target, epsilon=.1):
    # Calculate gradients for u1
    u1_perturbed = u1
    u1_perturbed += epsilon
    grad_u1 = (getfreq(u1_perturbed, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target)[0] - 
               getfreq(u1, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target)[0]) / epsilon
        
    # Calculate gradients for u2
    u2_perturbed = u2
    u2_perturbed += epsilon
    grad_u2 = (getfreq(u1, u2_perturbed, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target)[0] - 
               getfreq(u1, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target)[0]) / epsilon
    
    return grad_u1, grad_u2

def newtons_method(u1, u2, iterations, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target):
    lower_bound = 5
    upper_bound = 75
    max_delta = 2
    u1_list = []
    u2_list = []
    err_list = []
    update_list = []
    for i in range(iterations):
        grad = approximate_gradient(u1, u2,trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target)
        grad_array = np.array([grad[0], grad[1]])
        hessian = approximate_hessian(u1, u2, trainu_con, trainx_con, model, sc_x, sc_d, sc_i, target)


        # Ensure Hessian is invertible
        hessian_inv = np.linalg.inv(hessian)

        # Adaptive scaling factor 
        alpha = 1

        update = alpha * np.dot(hessian_inv, grad_array)  # This is now a vector of length 2
        
        # Update each parameter
        u1_update = u1 - update[0]
        u2_update = u2 - update[1]
        update_list.append(update)

        # Apply constraints
        if u1_update < lower_bound:
            u1_update = lower_bound
        elif u1_update > upper_bound:
            u1_update = upper_bound
        if u1_update > u1 + max_delta:
            u1_update = u1 + max_delta
        elif u1_update < u1 - max_delta:
            u1_update = u1 - max_delta
        if u2_update < lower_bound:
            u2_update = lower_bound
        elif u2_update > upper_bound:
            u2_update = upper_bound
        if u2_update > u2 + max_delta:
            u2_update = u2 + max_delta
        elif u2_update < u2 - max_delta:
            u2_update = u2 - max_delta

        # Update parameters
        u1 = u1_update
        u2 = u2_update
        err1, optimized_f_real = getfreq(u1, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i, target)
        err_list.append(err1)
        u1_list.append(u1)
        u2_list.append(u2)
        print('obj: '+str(err_list[i]))
    best_idx = np.argmin(err_list)
    return u1_list[best_idx], u2_list[best_idx],  update_list, u1_list, u2_list

def run_newtons_method(u1, u2, trainx_con, trainu_con, model, sc_x, sc_d, sc_i):
    target = np.zeros((1, 60))
    iterations = 1

    u1, u2, update_list, u1_list, u2_list = newtons_method(u1, u2, iterations, trainx_con, trainu_con, model, sc_x, sc_d, sc_i,  target)
    return u1, u2#, update_list, u1_list, u2_list