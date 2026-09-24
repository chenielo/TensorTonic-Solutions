import numpy as np

def loss_functions(y_true: list, y_pred: list, loss_type: str) -> float:
    """
    Returns the selected mean loss.
    """
    target = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if loss_type == "mse":
        value = np.mean((target-pred)**2)
    elif loss_type == "bce":
        prob = np.clip(pred, 1e-15, 1.0-1e-15)
        value =-np.mean(target*np.log(prob)+(1-target)*np.log(1.0-prob))
    elif loss_type == "cce":
        # 先做 softmax，把 logits 转成概率
        shifted = pred - np.max(pred, axis=1, keepdims=True)  # 数值稳定性技巧
        exp_pred = np.exp(shifted)
        prob = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        prob = np.clip(prob, 1e-15, 1.0-1e-15)
        
        # 再转 one-hot（如果 target 是类别索引）
        n_samples, n_classes = prob.shape
        y_indices = target.astype(int)
        target_oh = np.zeros((n_samples, n_classes))
        target_oh[np.arange(n_samples), y_indices] = 1.0
        
        value = -np.mean(np.sum(target_oh*np.log(prob), axis=1))
    elif loss_type=="hinge": 
        value=np.mean(np.maximum(0.0,1.0-target*pred))
    return round(float(value), 4)
