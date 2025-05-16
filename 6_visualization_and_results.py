from sklearn.metrics import classification_report
class_report = classification_report(y_true, y_pred_classes, target_names=["Healthy", "PD"])
print("Classification Report:\n", class_report)

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
y_test_binarized = label_binarize(y_test, classes=[0, 1])
y_pred_binarized = y_pred
fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_pred_binarized.ravel())
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

healthy_confidence = y_pred[y_true == 0][:, 0]
pd_confidence = y_pred[y_true == 1][:, 1]
healthy_mean, healthy_std = np.mean(healthy_confidence), np.std(healthy_confidence)
pd_mean, pd_std = np.mean(pd_confidence), np.std(pd_confidence)
print(f"Healthy Confidence - Mean: {healthy_mean:.2f}, Std Dev: {healthy_std:.2f}")
print(f"PD Confidence - Mean: {pd_mean:.2f}, Std Dev: {pd_std:.2f}")

from sklearn.metrics import cohen_kappa_score
kappa_score = cohen_kappa_score(y_true, y_pred_classes)
print(f"Cohen's Kappa Score: {kappa_score:.2f}")