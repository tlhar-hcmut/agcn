import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(df_confusion=None, file_name="confusion_matrix", title='Confusion matrix', cmap=plt.cm.gray_r):
    # print(df_confusion)
    # print(len(df_confusion.columns))
    # print(len(df_confusion.index))
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title,pad = 10)
    plt.colorbar()
    x_tick_marks = np.arange(len(df_confusion.columns))
    y_tick_marks = np.arange(len(df_confusion.index))
    plt.xticks(x_tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(y_tick_marks, df_confusion.index)
    #plt.tight_layout()
    
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
    plt.savefig(file_name)
    plt.close('all')