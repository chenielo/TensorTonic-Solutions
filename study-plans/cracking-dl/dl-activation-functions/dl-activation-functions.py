import numpy as np
def relu(x):
    out = max(0.0, x)
    deriv =1.0 if x > 0 else 0.0
    return out, deriv 
    
def leaky_relu(x):
    out = x if x > 0 else 0.01*x 
    deriv = 1.0 if x > 0 else 0.01 
    return out, deriv 
    
def sigmoid(x):
    out = 1.0/(1+np.exp(-x))
    deriv = out*(1-out)
    return out, deriv 
def tanh(x):
    ex = np.exp(x)
    nex = np.exp(-x)
    out = (ex - nex) / (ex + nex)
    deriv = 1-out**2
    return out, deriv 
    
def gelu(x):
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = float(np.tanh(inner))
    out = float(0.5 * x * (1 + t))
    sech2 = 1 - t ** 2
    inner_deriv = c * (1 + 3 * 0.044715 * x ** 2)
    deriv = float(0.5 * (1 + t) + 0.5 * x * sech2 * inner_deriv)
    return out, deriv 
def swish(x):
    sig = 1/(1+np.exp(-x))
    out = x*sig
    deriv = sig+x*sig*(1-sig)
    return out, deriv
def activation_functions(x, activation):
    """
    Returns: list
    """
    x = float(x)
    if activation == "relu":
        out, d = relu(x)
    elif activation == "leaky_relu":
         out, d = leaky_relu(x)
    elif activation == "tanh":
         out, d = tanh(x)
    elif activation == "sigmoid":
         out, d = sigmoid(x)
    elif activation == "gelu":
         out, d = gelu(x)            
    elif activation == "swish":
         out, d = swish(x)      
    return [round(out, 4), round(d, 4)]
    
