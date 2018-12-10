
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import sklearn



def plot_roc_curve_(model,y_pred,y_test):

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred[:, 1])

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label=model)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()






