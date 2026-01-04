import numpy as np

from gaussian_mixture import gaussian_mixture



N1=5000
N2=5000 
dim=10

mean1=[1,1]
cov1=[1,1]
mean2=[0,0]
cov2=[1,1]


train1= gaussian_mixture(dim, mean1, cov1)
train2= gaussian_mixture(dim, mean2, cov2)
X1= train1.generate(N1)
X2= train2.generate(N2)

#######compute KL
N_test=100000
X_test= train1.generate(N_test)

KL_true= np.mean(np.log(train1.value(X_test) )-  np.log(train2.value(X_test) )) 
print('true value',KL_true)

##### KDE estimate




#####################


 
######################

from NN import NumpyLogisticNN

model = NumpyLogisticNN(input_dim=dim, hidden=(32,32 ), l2=1e-4, seed=0)
model.fit(
    X1, X2, 
    epochs=100, batch_size=(N1+N2)//10, lr=2e-3)

temp=model.f_theta(X1)

print('NN estimate',np.mean(temp))


 
from NN_warm_init import NumpyLogisticNN
 

model = NumpyLogisticNN(input_dim=dim, hidden=(32,32), l2=1e-4, seed=0)
model.fit(
    X1, X2,
    epochs=100, batch_size=(N1+N2)//10, lr=2e-3,
    X_test=X1,         # enables drift-based early stop
    es_rel=0.01 ,         # 1%
    es_min_epochs=2,       # wait a couple epochs before checking
    es_norm_ord=2          # L2 norm over f(X_test)
)

temp=model.f_theta(X1)

print('NN estimate',np.mean(temp))

# 2) Create a target model for dataset B, initialized from the source
tgt = NumpyLogisticNN(input_dim=dim, hidden= (32,32), l2=1e-4,
                      init_from=None,   # copy all weights/biases
                      reset_last_layer=False)  # set True to randomize only the last layer

# 3) Fine-tune on dataset B
tgt.fit(
    X1, X2,
    epochs=50, batch_size=(N1+N2)//10, lr=2e-3,
    X_test=X1,         # enables drift-based early stop
    es_rel=0.01 ,         # 1%
    es_min_epochs=2,       # wait a couple epochs before checking
    es_norm_ord=2          # L2 norm over f(X_test)
)

temp=tgt.f_theta(X1)

print('NN estimate',np.mean(temp))


####################
if dim<20:

    from density_utility import kernel_density, KL_compute
    
    
    
    y1= kernel_density(X1).compute(X1)
    y2= kernel_density(X2).compute(X1)
    
    KL_KDE= KL_compute().compute(y1, y2)
    print('KDE', KL_KDE)
