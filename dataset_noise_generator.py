import numpy as np
import random 

def gen_noise(input_X,noise_coeff,noise_type='normal', seed=99, clean_nan=True, **kw_opts): 
#     TODO: noise_coeff as a vector -> different for each channel

    
    
    np.random.seed(seed)
    shape = input_X.shape
    
    if len(shape)==1:
        shape = (shape[0],1) 
    
    if noise_type=='normal':
        if 'loc' not in kw_opts.keys():
            loc = 0
        else:
            loc = kw_opts['loc']
        noise = np.random.normal(loc,noise_coeff,shape)

    elif noise_type=='normal_prop':
        if 'loc' not in kw_opts.keys():
            loc = 0
        else:
            loc = kw_opts['loc']
        f = lambda x: np.random.normal(loc,np.abs(x.reshape(shape)),shape)
        noise = noise_coeff*f(input_X)

    elif noise_type=='poisson':
        noise = np.random.poisson(lam=np.abs(input_X),size=None)
        
    elif noise_type=='sin':
        try: 
            n_pi = kw_opts['n_osc']*2
        except:
            n_pi = 2 
        samples = np.linspace(0, n_pi*np.pi, num=shape[0], endpoint=False)
        noise = noise_coeff*np.tile(np.sin(samples), (shape[1],1)).T
        
    elif noise_type=='sin_index': 
        try: 
            n_pi = kw_opts['n_osc']*2
        except:
            n_pi = 2 
        ind = np.argsort(input_X)
        ind_rev = np.argsort(ind)
        input_X_sort = np.sort(input_X)
        errors_sort = gen_noise(input_X_sort,noise_coeff,'sin',seed=seed,n_osc=n_pi)
        if 'func' in kw_opts.keys(): # apply transformation to the linearly generated errors 
            errors_sort = kw_opts['func'](errors_sort)
        noise = errors_sort[ind_rev]
        
    elif noise_type=='linear_inc': 
        samples = np.linspace(0, shape[0], num=shape[0], endpoint=False)
        if len(shape)>1: 
            noise = noise_coeff*np.tile(samples/shape[0], (shape[1],1)).T
        else: 
            noise = noise_coeff*(samples/shape[0])
            
    elif noise_type=='population_weights':
        random.seed(seed)
        noise = random.choices(population=kw_opts['population'], weights=kw_opts['weights'],k=shape[0])
        
    elif noise_type=='linear_inc_index': 
        ind = np.argsort(input_X)
        ind_rev = np.argsort(ind)
        input_X_sort = np.sort(input_X)
        errors_sort = gen_noise(input_X_sort,noise_coeff,'linear_inc',seed=seed)
        if 'func' in kw_opts.keys(): # apply transformation to the linearly generated errors 
            errors_sort = kw_opts['func'](errors_sort)
        noise = errors_sort[ind_rev]
        
    elif noise_type=='linear_prop':
        plusminus = gen_noise(input_X,1,noise_type="population_weights",population=[-1,1], weights=[50,50])
        if 'func' in kw_opts.keys(): 
            noise = np.multiply(plusminus,kw_opts['func'](input_X))
        else: 
            noise = np.multiply(plusminus,input_X)
        noise = noise/noise.max()
        noise = noise_coeff*noise
        
    elif noise_type=='cauchy': 
        # cauchy will generate some heavy outliers if not truncated
        s = np.random.standard_cauchy(size=shape[0])
        if 'truncate' in kw_opts.keys(): 
            truncate = kw_opts['truncate']
            s_ind = np.ones(s.shape)
            while s_ind.sum() > 0:
                print(s_ind.sum())
                s_ind = (s<-truncate) | (s>truncate)
                s[s_ind] = np.random.standard_cauchy(s_ind.sum())
        noise = noise_coeff*s 
    
    elif noise_type=='uniform':         
        if 'low' not in kw_opts.keys():
            low = 0
        else: 
            low = kw_opts['low']            
        if 'high' not in kw_opts.keys():
            high = 1
        else: 
            high = kw_opts['high']            
        noise = np.random.uniform(low=low, high=high, size=shape)
    
    elif noise_type=='lambda': 
        noise = kw_opts['func'](**kw_opts['func_args'])
        
    rem_nan = lambda x: np.nan_to_num(x, nan=np.random.normal(0,1), posinf=np.random.normal(0,1), neginf=np.random.normal(0,1))
    if clean_nan: 
        noise = np.vectorize(rem_nan)(noise)
    return np.squeeze(noise)    




def theoretical_function_linear(X, coeffs, bias=0, y_noise_coeff=0, noise_type='normal', noise_args={}):
    noise_y = gen_noise(X[:,0],[y_noise_coeff],noise_type,**noise_args)
    Y_base = (X*coeffs).sum(axis=1)+bias
    print('mean abs Y: ',(np.abs(Y_base)).mean())
    print('mean abs noise: ',(np.abs(noise_y)).mean())
    print('mean SNR: ',(np.abs(Y_base/noise_y)).mean())
    return Y_base+noise_y


import pandas as pd

def data_to_df(X, y):
    df = pd.DataFrame(X, columns=['x'+str(i) for i in range(X.shape[1])])
    df['y'] = y
    return df



from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement

def make_regression_custom(
    n_samples=100,
    n_features=100,
    *,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    effective_rank=None,
    tail_strength=0.5,
    noise=0.0,
    shuffle=True,
    coef=False,
    random_state=None,
    custom_coef=None,
):

    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    if effective_rank is None:
        # Randomly generate a well conditioned input set
        X = generator.randn(n_samples, n_features)

    else:
        # Randomly generate a low rank, fat tail input set
        X = make_low_rank_matrix(
            n_samples=n_samples,
            n_features=n_features,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            random_state=generator,
        )

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = np.zeros((n_features, n_targets))
    num_custom_coef = np.array(custom_coef).size
    if (custom_coef is None) or (num_custom_coef!=(n_informative*n_targets)):
        custom_coef = 100 * generator.rand(n_informative, n_targets)
        if num_custom_coef!=(n_informative*n_targets): 
            print("not all coefficients present, reverted to random coeffs")
    else: 
        custom_coef = np.reshape(custom_coef,(n_informative, n_targets))
            
    ground_truth[:n_informative, :] = custom_coef
    y = np.dot(X, ground_truth) + bias

    # Add noise
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    y = np.squeeze(y)

    if coef:
        return X, y, np.squeeze(ground_truth)

    else:
        return X, y



# https://stackoverflow.com/questions/19397719/could-numpy-random-poisson-be-used-to-add-poisson-noise-to-images
# https://stackoverflow.com/questions/51050658/how-to-generate-random-numbers-with-predefined-probability-distribution

# https://scikit-learn.org/stable/datasets/sample_generators.html#sample-generators