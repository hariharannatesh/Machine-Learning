import numpy as np
epochs=50000
input_size,hidden_layers,output_size=2,3,1
LR=0.1
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
W1=np.random.uniform(size=(input_size,hidden_layers))
W2=np.random.uniform(size=(hidden_layers,output_size))
print(W2.T)

def sigmoid(x):
    return (1/(1+np.exp(-x)))
def sigmoid_prime(x):
    return (sigmoid(x)*(1-sigmoid(x)))

for epoch in range(epochs):
    out_hidden=sigmoid(np.dot(X,W1))
    output=np.dot(out_hidden,W2)
    error=y-output
    if epoch%5000==0:
        print(sum(error))
    dz=error*LR
    W2=W2+out_hidden.T.dot(dz)
    dh=dz.dot(W2.T)*(sigmoid_prime(np.dot(X,W1)))
    W1=W1+X.T.dot(dh)

X_test=X[1]
out_hid=sigmoid(np.dot(X_test,W1))
out=np.dot(out_hid,W2)
print(out)
    

             
                     
