import numpy as np

from gaussian_mixture import gaussian_mixture


N_train=500
bb=600 
N_total=700
dim=10


mean1=[1,1]
cov1=[1,1]
mean2=[0,0]
cov2=[1,1]

train1= gaussian_mixture(dim, mean1, cov1)
train2= gaussian_mixture(dim, mean2, cov2)
data= train1.generate(N_train)

def generate_online(cur_data,cur_time,bb):
    if cur_time<bb:
        new_vec= train1.generate(1)
    else:
        new_vec= train2.generate(1)
    return np.concatenate((cur_data,  new_vec ), axis=0)





from NN_warm_init import NumpyLogisticNN

window_size=30
NN_tructure=(32,32)
init=[[] for __ in range(window_size)]
record=[]
import matplotlib.pyplot as plt    

for i in range(N_train, N_total):
    print(i)
    data=generate_online(data, i, bb)
    test_value=0
    for w in range( window_size//2, window_size ):
        if not init[w]:
            init_from=None
        else:
            init_from=init[w]
        model = NumpyLogisticNN(input_dim=dim, hidden=NN_tructure, init_from= init_from , l2=1e-4 )
        model.fit(
            data[0:(i-w)], data[(i-w):i],
            epochs=100, batch_size=i//10, lr=2e-3,
            X_test=data[0:N_train],         # enables drift-based early stop
            es_rel=0.01 ,         
            es_min_epochs=2,       # wait a couple epochs before checking
            es_norm_ord=2  # L2 norm over f(X_test)
        )
        init[w]=model
            
        test_value=max(test_value, np.mean(model.f_theta(data[0:N_train])))
        
    record.append(test_value)
#W0, B0=init[26].get_params()    
#record = np.array(record)
plt.plot(record)             # x = 0,1,2,3; line plot
plt.xlabel("index"); 
plt.ylabel("value"); 
plt.grid(True)
plt.show()