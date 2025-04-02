import numpy as np

def compute_qkv(X,W_q,W_k,W_v):
    Q = np.dot(X ,W_q)
    K = np.dot(X , W_k)
    V = np.dot(X ,W_v)
    return Q , K , V


def self_attention(Q, K, V): 
    d_k= K.shape[1]

    score = np.matmul(Q  , K.T) / np.sqrt(d_k)
    

    attention = np.exp(score)/np.sum(np.exp(score) , axis=1 , keepdims= True)

    attention_output = np.matmul(attention , V)




    
	return attention_output
