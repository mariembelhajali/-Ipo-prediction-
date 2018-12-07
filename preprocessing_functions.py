

import numpy as np
import pandas as pd

import re as re
import nltk
import datetime as dt
from datetime import datetime, timedelta
from sklearn.base import TransformerMixin
from nltk.stem import SnowballStemmer
import nltk.data
import nltk
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords


## Transform numerical columns
class DataFrameImputer(TransformerMixin):

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
def from_excel_ordinal(ordinal, _epoch0=datetime(1899, 12, 31)):
    if ordinal > 59:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)



def plot_roc_curve(model,y_pred):

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred[:, 1])

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_lr, tpr_lr, label=model)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


def replace_nan(serie, string):
    """
    replace nan in a data.series by the most frequent value
    (the input string is used to convert nan to np.nan)
    """
    # replace string by nan
    serie = serie.replace(string, np.nan)
    print('#nan :' + str(serie.isnull().values.sum()))
    # replace nan by the most frequent value
    serie = serie.fillna(serie.value_counts().idxmax())
    return serie

## Categorical columns with \n or / in row
def process_categorical_with_sep(ipo_processing,col,char,columns_to_drop):
    ipo_processing[col] = replace_nan(ipo_processing[col],'nan')

    ipo_processing = pd.concat([ipo_processing, ipo_processing[col].str.get_dummies(sep=char)], axis=1)
    columns_to_drop.append(col)

    return ipo_processing,columns_to_drop

def process_cat_columns(ipo_processing,col,columns_to_drop):
    ipo_processing[col] = replace_nan(ipo_processing[col],'nan')
    ipo_processing = pd.concat([ipo_processing,pd.get_dummies(ipo_processing[col])],axis = 1)
    columns_to_drop.append(col)
    return ipo_processing,columns_to_drop

def process_text_columns(risks):
    #remove terms with digits-----------------------------------------------------------------------
    risks = risks.apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))  

    #lower case-----------------------------------------------------------------------------
    # Lower case and separate into tokens

    risks = risks.apply(lambda x: x.lower().split())  

    #stop word removal--------------------------------------------------------------------------

    # Download stop words dataset of NLTK library
    nltk.download('stopwords')
    # Remove stop words

    stop_words = stopwords.words('english')


    risks=risks.apply(lambda x: [w for w in x if w not in stop_words])

    #stemming-------------------------------------------------------------------------------------------
    # Stem words

    stemmer = SnowballStemmer('english')

    risks_words=risks.apply(lambda x: [stemmer.stem(w) for w in x])

    # WE regroup every row of text to one string

    risks=risks.apply(lambda x: ' '.join(x))
    return risks,risks_words
