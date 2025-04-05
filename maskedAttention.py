import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
        #print(K.shape[-1])
	    score = np.matmul(Q, K.T) / np.sqrt(K.shape[1])
		score = score + mask
		exp_score = np.exp(score-np.max(score, axis=-1, keepdims=True))
        attention=exp_score /  (np.sum(exp_score, axis=-1, keepdims=True) )
		return np.matmul(attention , V)
