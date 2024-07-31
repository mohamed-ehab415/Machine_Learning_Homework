import numpy as np

""""
Here's the text in an easy-to-copy format:

When \( p_t \) is high: \( (1 - p_t)^\gamma \) is very small, reducing the loss contribution and minimizing the focus on easy examples.

When \( p_t \) is low: \( (1 - p_t)^\gamma \) is large, increasing the loss contribution and focusing more on hard examples.
"""


def focal_loss_with_class_weight(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Ensure the prediction is within the range [eps, 1-eps] to avoid log(0)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    focal_loss_pos = -alpha * np.power(1 - y_pred, gamma) * np.log(y_pred)
    focal_loss_neg = -(1 - alpha) * np.power(y_pred, gamma) * np.log(1 - y_pred)

    focal_los=np.where(y_true==1,focal_loss_pos,focal_loss_neg)
    return np.mean(focal_los)
y_true = np.array([0, 1, 1, 0, 1])  # true labels
y_pred = np.array([0.1, 0.8, 0.9, 0.3, 0.7])  # predicted probabilities for the positive class

loss_value = focal_loss_with_class_weight(y_true, y_pred, alpha=0.25, gamma=2)
print(f"Focal Loss: {loss_value}")
