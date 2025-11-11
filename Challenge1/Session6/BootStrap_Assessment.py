# USAGE
# python train_simple.py --fonts input/example_fonts --char-classifier output/simple_char.cpickle \
#	--digit-classifier output/simple_digit.cpickle

##### PYTHON PACKAGES
# Generic
import pickle
import cv2
import imutils
import numpy as np
import pandas
import os
from matplotlib import pyplot as plt
import scipy

# Classifiers
# include differnet classifiers
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support,precision_score, recall_score, f1_score, accuracy_score, classification_report

# OWN FUNCTIONS (MODIFY ACORDING TO YOUR LOCAL PATH)

aucSVC, aucKNN, aucMLP = [], [], []
accSVC, accKNN, accMLP = [], [], []
recSVC_micro, recKNN_micro, recMLP_micro = [], [], []


#### STEP0. EXP-SET UP

# DB Main Folder (MODIFY ACORDING TO YOUR LOCAL PATH)
ResultsDir=r'C:\Users\suzan\Desktop\ActividadVision\A6\Descriptors'
# Load Font DataSets
fileout=os.path.join(ResultsDir,'AlphabetDescriptors')+'.pkl'    
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()  
alphabetFeat=data['alphabetFeat']
alphabetLabels=data['alphabetLabels']
   

fileout=os.path.join(ResultsDir,'DigitsDescriptors')+'.pkl'   
f=open(fileout, 'rb')
data=pickle.load(f)
f.close()   
digitsFeat=data['digitsFeat']
digitsLabels=data['digitsLabels']




#### DEFINE BINARY DATASET
DescriptorsTags=list(digitsFeat.keys())
targetFeat=DescriptorsTags[0]

digits=np.stack(digitsFeat[targetFeat])
digitsLab=np.zeros(digits.shape[0])
chars=np.stack(alphabetFeat[targetFeat])
charsLab=np.ones(chars.shape[0])

X=np.concatenate((digits,chars))
y=np.concatenate((digitsLab,charsLab))

### STEP1. TRAIN BINARY CLASSIFIERS [CHARACTER VS DIGITS]
NTrial=50
aucMLP=[]
aucSVC=[]
aucKNN=[]
recMLP=[]
recSVC=[]
recKNN=[]
for kTrial in np.arange(NTrial):
    # Random Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)
    
    ##### SVM
    ## Train Model
    ModelSVC = SVC(C=1.0,class_weight='balanced') #compute loss with weights accounting for class unbalancing
    # Use CalibratedClassifierCV to calibrate probabilites
    ModelSVC = CalibratedClassifierCV(ModelSVC,n_jobs=-1)
    ModelSVC.fit(X_train, y_train)
    ## Evaluate Model
    pSVC = ModelSVC.predict_proba(X_test)
    
    ## Metrics
    auc=roc_auc_score(y_test, pSVC[:,1])
    aucSVC.append(auc)
    y_pred=(pSVC[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    recSVC.append(rec)
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    # Example shown for SVC after you form y_pred:
    prec_micro  = precision_score(y_test, y_pred, average='micro', zero_division=0)
    rec_micro   = recall_score(y_test,  y_pred, average='micro', zero_division=0)
    prec_macro  = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro   = recall_score(y_test,  y_pred, average='macro', zero_division=0)
    prec_weight = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weight  = recall_score(y_test,  y_pred, average='weighted', zero_division=0)
    acc         = accuracy_score(y_test, y_pred)
    
    print("[SVC] micro -> P:%.3f R:%.3f Acc:%.3f"   % (prec_micro, rec_micro, acc))
    print("[SVC] macro -> P:%.3f R:%.3f Acc:%.3f"   % (prec_macro, rec_macro, acc))
    print("[SVC] weighted -> P:%.3f R:%.3f Acc:%.3f" % (prec_weight, rec_weight, acc))
    
    from sklearn.metrics import precision_recall_fscore_support
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    print("[SVC] per-class precision:", prec_c, "recall:", rec_c, "support:", sup_c)
    
    # Compute simple means across classes (same as macro averages)
    prec_avg = np.mean(prec_c)
    rec_avg  = np.mean(rec_c)
    print("[SVC] mean per-class -> P:%.3f R:%.3f" % (prec_avg, rec_avg))
    
    
    # inside the for kTrial loop, after y_pred:
    accSVC.append(accuracy_score(y_test, y_pred))
    recSVC_micro.append(recall_score(y_test, y_pred, average='micro', zero_division=0))
    # (do same for KNN, MLP)



    ##### KNN
    ## Train Model
    ModelKNN = KNeighborsClassifier(n_neighbors=10)
    ModelKNN = CalibratedClassifierCV(ModelKNN,n_jobs=-1)
    ModelKNN.fit(X_train, y_train)
    ## Evaluate Model
    pKNN = ModelKNN.predict_proba(X_test)
    # Metrics
    auc=roc_auc_score(y_test, pKNN[:,1])
    aucKNN.append(auc)
    y_pred=(pKNN[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    recKNN.append(rec)
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    # Example shown for SVC after you form y_pred:
    prec_micro  = precision_score(y_test, y_pred, average='micro', zero_division=0)
    rec_micro   = recall_score(y_test,  y_pred, average='micro', zero_division=0)
    prec_macro  = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro   = recall_score(y_test,  y_pred, average='macro', zero_division=0)
    prec_weight = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weight  = recall_score(y_test,  y_pred, average='weighted', zero_division=0)
    acc         = accuracy_score(y_test, y_pred)
    
    print("[KNN] micro -> P:%.3f R:%.3f Acc:%.3f"   % (prec_micro, rec_micro, acc))
    print("[KNN] macro -> P:%.3f R:%.3f Acc:%.3f"   % (prec_macro, rec_macro, acc))
    print("[KNN] weighted -> P:%.3f R:%.3f Acc:%.3f" % (prec_weight, rec_weight, acc))
    
    from sklearn.metrics import precision_recall_fscore_support
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    print("[KNN] per-class precision:", prec_c, "recall:", rec_c, "support:", sup_c)
    
    # Compute simple means across classes (same as macro averages)
    prec_avg = np.mean(prec_c)
    rec_avg  = np.mean(rec_c)
    print("[KNN] mean per-class -> P:%.3f R:%.3f" % (prec_avg, rec_avg))
    

    
    accSVC.append(accuracy_score(y_test, y_pred))
    recSVC_micro.append(recall_score(y_test, y_pred, average='micro', zero_division=0))


    #### MLP
    ## Train Model
    ModelMLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15, 20), random_state=1,max_iter=100000)
    ModelMLP = CalibratedClassifierCV(ModelMLP,n_jobs=-1)
    ModelMLP.fit(X_train, y_train)
    ## Evaluate Model
    pMLP = ModelMLP.predict_proba(X_test)
    # Metrics
    auc=roc_auc_score(y_test, pMLP[:,1])
    aucMLP.append(auc)
    y_pred=(pMLP[:,1]>=0.5).astype(int)
    prec,rec,_,_ = precision_recall_fscore_support(y_test, y_pred,
                                       zero_division=0)
    recMLP.append(rec)
    
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    # Example shown for SVC after you form y_pred:
    prec_micro  = precision_score(y_test, y_pred, average='micro', zero_division=0)
    rec_micro   = recall_score(y_test,  y_pred, average='micro', zero_division=0)
    prec_macro  = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro   = recall_score(y_test,  y_pred, average='macro', zero_division=0)
    prec_weight = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weight  = recall_score(y_test,  y_pred, average='weighted', zero_division=0)
    acc         = accuracy_score(y_test, y_pred)
    
    print("[MLP] micro -> P:%.3f R:%.3f Acc:%.3f"   % (prec_micro, rec_micro, acc))
    print("[MLP] macro -> P:%.3f R:%.3f Acc:%.3f"   % (prec_macro, rec_macro, acc))
    print("[MLP] weighted -> P:%.3f R:%.3f Acc:%.3f" % (prec_weight, rec_weight, acc))
    
    from sklearn.metrics import precision_recall_fscore_support
    prec_c, rec_c, f1_c, sup_c = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    print("[MLP] per-class precision:", prec_c, "recall:", rec_c, "support:", sup_c)
    
    # Compute simple means across classes (same as macro averages)
    prec_avg = np.mean(prec_c)
    rec_avg  = np.mean(rec_c)
    print("[MLP] mean per-class -> P:%.3f R:%.3f" % (prec_avg, rec_avg))
    

    
    accSVC.append(accuracy_score(y_test, y_pred))
    recSVC_micro.append(recall_score(y_test, y_pred, average='micro', zero_division=0))


import pandas as pd
def to_series(x): return pd.Series(x)

plt.figure(); plt.boxplot([aucSVC, aucKNN, aucMLP], labels=['SVM','KNN','MLP']); plt.title("AUC")
plt.figure(); plt.boxplot([accSVC, accKNN, accMLP], labels=['SVM','KNN','MLP']); plt.title("Accuracy")
plt.figure(); plt.boxplot([recSVC_micro, recKNN_micro, recMLP_micro], labels=['SVM','KNN','MLP']); plt.title("Recall (micro)")

plt.figure(); plt.hist(aucSVC, bins=10, alpha=0.6, label='SVM')
plt.hist(aucKNN, bins=10, alpha=0.6, label='KNN')
plt.hist(aucMLP, bins=10, alpha=0.6, label='MLP'); plt.title("AUC histograms"); plt.legend()

# Bar of means
means = [np.mean(aucSVC), np.mean(aucKNN), np.mean(aucMLP)]
plt.figure(); plt.bar(['SVM','KNN','MLP'], means); plt.ylabel("Mean AUC"); plt.title("Mean AUC by classifier")
plt.show()

#### STEP2. ANALYZE RESULTS
## Visual Exploration
recSVC=np.stack(recSVC)
recKNN=np.stack(recKNN)
recMLP=np.stack(recMLP)
# #### Plots accoss trials (random splits)
plt.figure()
plt.plot(np.arange(NTrial),aucSVC,marker='o',c='b',markersize=10)
plt.plot(np.arange(NTrial),aucKNN,marker='o',c='r',markersize=10)
plt.plot(np.arange(NTrial),aucMLP,marker='o',c='g',markersize=10)
plt.legend(['SVM','KNN','MLP'])
plt.xticks(np.arange(NTrial), fontsize=10)
plt.xlabel("Trial", fontsize=15)
plt.ylabel("AUC", fontsize=15)
plt.show()

def describe(x):
    x = np.asarray(x)
    q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                std=float(x.std(ddof=1)), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4],
                min=float(x.min()), max=float(x.max()))

print("AUC SVM:", describe(aucSVC))
print("AUC KNN:", describe(aucKNN))
print("AUC MLP:", describe(aucMLP))


import scipy.stats as st

def ci95(x):
    x = np.asarray(x, dtype=float)
    return st.t.interval(confidence=0.95, df=len(x)-1,
                         loc=np.mean(x), scale=st.sem(x, ddof=1))

print("CI AUC SVM:", ci95(aucSVC))
print("CI AUC KNN:", ci95(aucKNN))
print("CI AUC MLP:", ci95(aucMLP))

# Example for AUC: SVM vs KNN
diff_SVM_KNN = np.asarray(aucSVC) - np.asarray(aucKNN)
tval, pval = st.ttest_1samp(diff_SVM_KNN, 0.0)
print("AUC (SVM-KNN) t=%.3f  p=%.4g" % (tval, pval))

# Repeat all pairings and for other metrics (accuracy, micro recall)
def pair_test(a, b, name):
    d = np.asarray(a) - np.asarray(b)
    t, p = st.ttest_1samp(d, 0.0)
    print(f"{name}: t={t:.3f} p={p:.4g}")

pair_test(aucSVC, aucMLP, "AUC SVM-MLP")
pair_test(aucKNN, aucMLP, "AUC KNN-MLP")

