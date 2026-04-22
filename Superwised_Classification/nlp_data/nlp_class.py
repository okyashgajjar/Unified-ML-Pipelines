import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
from scipy.sparse import hstack, csr_matrix

# ----- Models -----
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# ----- Pipeline & Selection -----
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ----- Metrics -----
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score

# ----- Vectorization ----
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None

from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore")

stop_words = set([
    'the', 'is', 'in', 'and', 'to', 'with', 'a', 'an', 'of', 'for', 'on', 'at',
    'by', 'this', 'that', 'are', 'was', 'it', 'be', 'as', 'from', 'or', 'has',
    'have', 'had', 'but', 'not', 'he', 'she', 'they', 'you', 'we', 'his', 'her'
])

def clean_data(text):
    if not isinstance(text, str):
        return ""
    Cltext = re.sub('https\S+', ' ', text) #links
    Cltext = re.sub('@', ' ', Cltext) #mentions
    Cltext = re.sub('\n', ' ', Cltext) #newline
    Cltext = re.sub('\r', ' ' , Cltext) #carriage returns
    Cltext = re.sub(r'[^\w\s]', ' ', Cltext) #Punctuations
    Cltext = re.sub(r"[\u2600-\u26FF\u2700-\u27BF]+", ' ', Cltext) #Emoji ranges
    Cltext = re.sub('\s+', ' ', Cltext).strip() #extra whitespace
    Cltext = re.sub('_', ' ', Cltext)
    Cltext = re.sub("'", ' ', Cltext)

    #Lowercase from consistent case removal
    Cltext = Cltext.lower()

    words = Cltext.split()
    cleaned_words = [word for word in words if word not in stop_words]

    return ' '.join(cleaned_words)

# ── Vectorizers 
class NLPVectorizers:
    def __init__(self, w2v_size=200, w2v_window=5):
        self.tfidf = TfidfVectorizer(
            ngram_range=(1,1), max_features=50000,
            sublinear_tf=True, min_df=2, max_df=0.95
        )
        self.tfidf_ng = TfidfVectorizer(
            ngram_range=(1,2), max_features=100000,
            sublinear_tf=True, min_df=3, max_df=0.90
        )
        self.w2v_size = w2v_size
        self.w2v_window = w2v_window
        self.w2v_model = None
        self.scaler = StandardScaler()

    def fit(self, texts):
        tokenized = [t.split() for t in texts]
        self.tfidf.fit(texts)
        self.tfidf_ng.fit(texts)
        if Word2Vec:
            self.w2v_model = Word2Vec(
                tokenized, vector_size=self.w2v_size,
                window=self.w2v_window, min_count=2,
                workers=4, epochs=10
            )
            X_w2v = self._avg_w2v(tokenized)
            self.scaler.fit(X_w2v)
        return self

    def transform(self, texts):
        tokenized = [t.split() for t in texts]
        X_tfidf     = self.tfidf.transform(texts)
        X_tfidf_ng  = self.tfidf_ng.transform(texts)
        
        results = {
            "tfidf":    X_tfidf,       # → Linear, Ridge, Lasso, Logistic, SVC-linear
            "tfidf_ng": X_tfidf_ng,    # → SVC, Ridge Classif
        }

        if self.w2v_model:
            X_w2v_raw   = self._avg_w2v(tokenized)
            X_w2v       = self.scaler.transform(X_w2v_raw)
            # concat = sparse tfidf + dense w2v (as sparse)
            X_concat = hstack([X_tfidf_ng, csr_matrix(X_w2v)])
            results["w2v"] = X_w2v
            results["concat"] = X_concat
        else:
            # Fallback if w2v is not available
            results["w2v"] = X_tfidf
            results["concat"] = X_tfidf_ng
            
        return results

    def _avg_w2v(self, tokenized):
        out = []
        if not self.w2v_model:
            return np.zeros((len(tokenized), self.w2v_size))
        for tokens in tokenized:
            vecs = [self.w2v_model.wv[w]
                    for w in tokens if w in self.w2v_model.wv]
            out.append(np.mean(vecs, axis=0)
                       if vecs else np.zeros(self.w2v_size))
        return np.array(out)

# Classification pipeline 

def build_classifiers():
    models = {
        # Weight-Based → tfidf
        "logistic":      (LogisticRegression(C=1.0, max_iter=1000,
                           solver='lbfgs'), "tfidf"),
        "ridge_clf":     (RidgeClassifier(alpha=1.0), "tfidf"),

        # Kernel-Based → tfidf_ng / w2v
        "svc_linear":    (CalibratedClassifierCV(
                           LinearSVC(C=0.5, max_iter=2000)), "tfidf_ng"),
        "svc_rbf":       (SVC(kernel='rbf', C=1.0,
                           probability=True), "w2v"),
        "svc_poly":      (SVC(kernel='poly', degree=3,
                           probability=True), "w2v"),

        # Tree-Based → w2v or concat
        "dt":            (DecisionTreeClassifier(max_depth=10), "w2v"),
        "rf":            (RandomForestClassifier(n_estimators=200,
                           n_jobs=-1), "w2v"),

        # Neural Net → w2v scaled
        "mlp_clf":       (MLPClassifier(hidden_layer_sizes=(256,128),
                           activation='relu', max_iter=100,
                           early_stopping=True), "w2v"),

        # Instance-Based → w2v
        "knn_clf":       (KNeighborsClassifier(n_neighbors=5,
                           metric='cosine'), "w2v"),
    }
    
    # Optional models
    # if XGBClassifier:
    #     models["xgb"] = (XGBClassifier(n_estimators=300, max_depth=6,
    #                        learning_rate=0.1, tree_method='hist',
    #                        eval_metric='mlogloss'), "concat")
    # if LGBMClassifier:
    #     models["lgbm"] = (LGBMClassifier(n_estimators=300,
    #                        learning_rate=0.1, n_jobs=-1), "concat")
                           
    return models

def nlp_classification(df, target_column, text_column=None):
    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
            
        if text_column is None:
            # Find the best candidate for text column (object type with highest average string length)
            candidates = df.select_dtypes(include=['object']).columns.tolist()
            candidates = [c for c in candidates if c != target_column]
            if not candidates:
                # If no object columns, try all columns except target
                candidates = [c for c in df.columns if c != target_column]
            
            if not candidates:
                raise ValueError("No text column found in dataframe.")
            
            # Select column with max average length as the likely text column
            text_column = max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
            print(f"Auto-detected text column: '{text_column}'")

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataframe")
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Clean data
        df['cleaned_input'] = df[text_column].apply(lambda x: clean_data(x))
        
        texts = df['cleaned_input']
        y = df[target_column]
        
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            texts, y, test_size=0.2, random_state=42
        )

        # fit vectorizers on train only
        vec = NLPVectorizers()
        vec.fit(X_train_raw)
        X_train = vec.transform(X_train_raw)
        X_test  = vec.transform(X_test_raw)

        models_dict = build_classifiers()
        all_results = []

        for model_name, (model, feat_key) in models_dict.items():
            try:
                Xtr = X_train[feat_key]
                Xte = X_test[feat_key]

                # convert sparse to dense for models that need it
                if feat_key == "w2v" and hasattr(Xtr, "toarray"):
                    Xtr, Xte = Xtr.toarray(), Xte.toarray()

                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)

                acc_score = accuracy_score(y_test, y_pred)
                prec_score = precision_score(y_test, y_pred, average='weighted')
                f1_val = f1_score(y_test, y_pred, average='weighted')

                all_results.append({
                    'model_name': model_name,
                    'best_params': model.get_params() if hasattr(model, 'get_params') else None,
                    'pipeline': model,
                    'accuracy': acc_score,
                    'precision': prec_score,
                    'f1_score': f1_val,
                    'status': 'success',
                    'error': None
                })
            except Exception as model_error:
                all_results.append({
                    'model_name': model_name,
                    'best_params': None,
                    'pipeline': None,
                    'accuracy': None,
                    'precision': None,
                    'f1_score': None,
                    'status': 'failed',
                    'error': str(model_error)
                })

        return pd.DataFrame(all_results)

    except Exception as e:
        return pd.DataFrame([{
            'model_name': 'NLP Classification Pipeline',
            'best_params': None,
            'pipeline': None,
            'accuracy': None,
            'precision': None,
            'f1_score': None,
            'status': 'failed',
            'error': f"NLP pipeline failed: {str(e)}"
        }])
