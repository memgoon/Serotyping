#!/usr/bin/env python
# coding: utf-8

# # ML classification of E.coli serotype using Large Language Model

# ### 1. Import Packages

# In[92]:


import numpy as np
import re
from IPython.lib.deepreload import reload
import pandas as pd
import logomaker as lm
import os, glob, subprocess, math, pickle, time, copy, seqlogo, json
from dna_features_viewer import GraphicFeature, GraphicRecord
from sklearn.metrics import classification_report
import matplotlib.gridspec as gridspec
from Bio import SeqIO, AlignIO, SearchIO
from Bio.Align import MultipleSeqAlignment
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
from multiprocessing import Pool
from subprocess import Popen
from dna_features_viewer import BiopythonTranslator
from dna_features_viewer import GraphicFeature, GraphicRecord
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from multiscorer.multiscorer import MultiScorer
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn import tree
from sklearn import neighbors
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso, LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import entropy
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from matplotlib import rcParams
from sklearnex import patch_sklearn
from sklearn.model_selection import StratifiedShuffleSplit


# In[2]:


patch_sklearn()
rcParams['figure.figsize'] = 8,8


# ### 2. To use MBGD ortholog definition, import the parsed data file

# In[3]:


with open("2-1.split/GeneMeta.pickle", "rb") as infile:
    GeneMeta = pickle.load(infile)
with open("2-1.split/Gene2Cluster.pickle", "rb") as infile:
    Gene2Cluster = pickle.load(infile)
with open("2-1.split/ClusterMeta.pickle", "rb") as infile:
    ClusterMeta = pickle.load(infile)


# ### 3. Get the real serotype annotation information

# In[4]:


meta = pd.read_csv("meta.csv", sep="\t", header=None)
meta.columns = ["acc", "O", "H"]
meta


# In[5]:


meta_name = pd.read_csv("meta_name.csv", sep="\t", header=None)
meta_name.columns = ["entero_acc", "NCBI_acc", "PRJNA"]
meta_name


# ### 4. Filter out Mash Failed Samples 

# In[6]:


ms_res = pd.read_csv("1-1.mash/mash_output.txt", sep="\t", header=None)
ms_res.columns = ["v1", "v2", "sample_name", "mash_dist"]
ms_res


# In[7]:


filter_acc = ms_res[ms_res["mash_dist"] > 0.04]
len(filter_acc)


# In[8]:


plt.hist(ms_res["mash_dist"])


# In[9]:


meta = meta[~meta["acc"].isin(list(filter_acc["sample_name"]))]
meta_name = meta_name[meta_name["entero_acc"].isin(list(meta["acc"]))]
meta = pd.merge(meta, meta_name, left_on="acc", right_on="entero_acc", how="inner")
meta = meta.drop(["entero_acc"], axis=1)
meta


# In[10]:


meta.to_csv("final_samples.csv")


# In[11]:


OHtable = meta.set_index("acc").T.to_dict("list")
OHtable


# ###  5. Plot Data Distribution according to O types

# In[12]:


#to add percentages
def my_autopct(pct):
    return ('%1.1f%%' % pct) if pct >= 1.2 else ''

plt.pie(meta["O"].value_counts(),
        labels=["{} (n={})".format(name, meta["O"].value_counts()[name]) if idx <12 else "" for idx, name in enumerate(list(meta["O"].value_counts().keys()))],
        autopct=my_autopct,
        pctdistance=.85,
        explode=[0.1]*len(meta["O"].value_counts()),
        shadow=True,
        colors=plt.cm.Pastel1.colors
       )
plt.show() 


# In[13]:


major_O_list = [k for k, v in dict(meta["O"].value_counts()).items() if v>=5]
len(major_O_list) ## Number of O classes having more than 5 data


# In[14]:


sum([v for k, v in dict(meta["O"].value_counts()).items() if v>=10])


# ### 6. Get Prediction Results from Conventional Tools: SerotypeFinder and ECTyper

# #### SerotypeFinder

# In[15]:


sfTable = pd.read_csv("Eco_over1kb/serotypeFinder/result.txt", sep="\t")
sfTable = sfTable[sfTable["file"].isin(list(meta["acc"]))]


# In[16]:


sfTable


# In[17]:


"{}/{} ({:.2f}%) Entries were ambiguous O type in SerotypeFinder".format((sfTable["O_tag"] == "ambiguous").sum(),
                                                                         len(sfTable),
                                                                         (sfTable["O_tag"] == "ambiguous").sum()/len(sfTable))


# In[18]:


"{}/{} ({:.2f}%) Entries could not determine O type in SerotypeFinder".format(len(sfTable) - sum(sfTable["O_type"].value_counts()),
                                                                       len(sfTable),
                                                                       100*(len(sfTable) - sum(sfTable["O_type"].value_counts()))/len(sfTable)
                                                                      )


# In[19]:


compareTb = pd.merge(meta, sfTable, left_on="acc", right_on="file", how="inner")
compareTb


# In[20]:


compareTb[compareTb["O"] == "O157"]


# In[21]:


sf_report = pd.DataFrame(classification_report(list(compareTb["O"]), list(compareTb["O_type"]), output_dict=True)).transpose()


# In[22]:


sf_report.loc["O157"]


# In[23]:


list(dict(meta["O"].value_counts()).values())[100:110]


# In[24]:


confusion_matrix(list(compareTb["O"]), list(compareTb["O_type"]), labels=["O157"])


# In[26]:


major_O_list = list(dict(meta["O"].value_counts()).keys())[0:5]
middle_O_list = list(dict(meta["O"].value_counts()).keys())[120:125]
major_cm = confusion_matrix(list(compareTb["O"]), list(compareTb["O_type"]), labels=major_O_list)
middle_cm = confusion_matrix(list(compareTb["O"]), list(compareTb["O_type"]), labels=middle_O_list)

major_O_count_sums = np.array(list(dict(meta["O"].value_counts()).values())[0:5])
middle_O_count_sums = np.array(list(dict(meta["O"].value_counts()).values())[120:125])
major_cm_rt = major_cm / major_O_count_sums[:, np.newaxis]
middle_cm_rt = middle_cm / middle_O_count_sums[:, np.newaxis]

# 혼동 행렬을 시각화하는 함수
def plot_confusion_matrix(ax, cm, labels, sum_value, title, ylabel, xlabel):
    #plt.figure(figsize=(6, 4))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=8)
    #plt.colorbar(cax, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=8)

    # 텍스트 표시
    fmt = ""#'.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j] if i!=j else str(cm[i, j])+"\n("+str(sum_value[i])+")", fmt),
                 ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    #plt.tight_layout()

# 레이블 예시
labels = ['Class 1', 'Class 2', 'Class 3']

# 혼동 행렬 플롯
fig, ax = plt.subplots(2, 1, figsize=(2, 5))
#plot_confusion_matrix(ax[0], major_cm_rt, major_O_list, "Recall for SerotypeFinder\nMajor O types")
plot_confusion_matrix(ax[0], major_cm, major_O_list, major_O_count_sums, "Confusion Matrix for SerotypeFinder\n(Major O types)", "True label", "")
#plot_confusion_matrix(ax[1], middle_cm_rt, middle_O_list, "Recall for SerotypeFinder\nMinor O types")
plot_confusion_matrix(ax[1], middle_cm, middle_O_list, middle_O_count_sums, "(Minor O types)", "True label", "Predicted label")
plt.show()


# In[27]:


"By SerotypeFinder, {}/{} ({:.2f}%) O types were correctly classified".format((compareTb["O"] == compareTb["O_type"]).sum(),
                                                                              len(compareTb),
                                                                              100*(compareTb["O"] == compareTb["O_type"]).sum()/len(compareTb),                                                                               )


# In[28]:


"By SerotypeFinder, {}/{} ({:.2f}%) H types were correctly classified".format((compareTb["H"] == compareTb["H_type"]).sum(),
                                                                              len(compareTb),
                                                                              100*(compareTb["H"] == compareTb["H_type"]).sum()/len(compareTb),                                                                               )


# #### ECTyper

# In[29]:


ectyper_stat = pd.read_csv("Eco_over1kb/ectyper_temp2/output.tsv", sep="\t")
ectyper_stat = ectyper_stat[~ectyper_stat["Warnings"].str.contains("Non fasta")]
ectyper_stat["Name"] = list(ectyper_stat.Name.str.split(".").str[0])
ectyper_merged = pd.merge(left=ectyper_stat, right=meta, how="left", left_on="Name", right_on="acc")
ectyper_merged
ectyper_merged


# In[30]:


ectyper_merged[ectyper_merged["O-type"].str.contains("/")]


# In[31]:


"{}/{} ({:.2f}%) Entries were ambiguous O type in ECTyper".format((ectyper_merged["O-type"].str.contains("/")).sum(),
                                                                         len(ectyper_merged),
                                                                         100*(ectyper_merged["O-type"].str.contains("/")).sum()/len(ectyper_merged))


# In[32]:


"{}/{} ({:.2f}%) Entries could not determine O type in ECTyper".format((ectyper_merged["O-type"] == "-").sum(),
                                                                         len(ectyper_merged),
                                                                         100*(ectyper_merged["O-type"] == "-").sum()/len(ectyper_merged))


# In[33]:


"By ECTyper, {}/{} ({:.2f}%) O types were correctly classified".format(len(ectyper_merged[ectyper_merged["O-type"] == ectyper_merged["O"]]),
                                                                              len(ectyper_merged),
                                                                              100*len(ectyper_merged[ectyper_merged["O-type"] == ectyper_merged["O"]])/len(ectyper_merged),)


# In[34]:


"By ECTyper, {}/{} ({:.2f}%) H types were correctly classified".format(len(ectyper_merged[ectyper_merged["H-type"] == ectyper_merged["H"]]),
                                                                              len(ectyper_merged),
                                                                              100*len(ectyper_merged[ectyper_merged["H-type"] == ectyper_merged["H"]])/len(ectyper_merged),)


# In[35]:


ec_report = pd.DataFrame(classification_report(list(ectyper_merged["O"]), list(ectyper_merged["O-type"]), output_dict=True)).transpose()


# In[36]:


ec_report


# In[37]:


major_O_list = list(dict(meta["O"].value_counts()).keys())[0:5]
middle_O_list = list(dict(meta["O"].value_counts()).keys())[100:105]
major_cm = metrics.confusion_matrix(list(ectyper_merged["O"]), list(ectyper_merged["O-type"]), labels=major_O_list)
middle_cm = metrics.confusion_matrix(list(ectyper_merged["O"]), list(ectyper_merged["O-type"]), labels=middle_O_list)

major_O_count_sums = np.array(list(dict(meta["O"].value_counts()).values())[0:5])
middle_O_count_sums = np.array(list(dict(meta["O"].value_counts()).values())[100:105])
major_cm_rt = major_cm / major_O_count_sums[:, np.newaxis]
middle_cm_rt = middle_cm / middle_O_count_sums[:, np.newaxis]

# 혼동 행렬을 시각화하는 함수
def plot_confusion_matrix(ax, cm, labels, title, ylabel, xlabel):
    #plt.figure(figsize=(6, 4))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=8)
    #plt.colorbar(cax, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=8)

    # 텍스트 표시
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    #plt.tight_layout()

# 레이블 예시
labels = ['Class 1', 'Class 2', 'Class 3']

# 혼동 행렬 플롯
fig, ax = plt.subplots(2, 1, figsize=(2, 5))
#plot_confusion_matrix(ax[0], major_cm_rt, major_O_list, "Recall for ECTyper\nMajor O types")
plot_confusion_matrix(ax[0], major_cm_rt, major_O_list, "Recall for ECTyper\n(Major O types)", "True label", "")
#plot_confusion_matrix(ax[1], middle_cm_rt, middle_O_list, "Recall for ECTyper\nMinor O types")
plot_confusion_matrix(ax[1], middle_cm_rt, middle_O_list, "(Minor O types)", "True label", "Predicted label")
plt.show()


# ### 7. Parse GeneMark-S2 Results and MBGD Blast Results (DO NOT RUN AGAIN; takes long time to process)

# #### Parse GeneMark-S2 Results

# In[117]:


fl = glob.glob("2-1.split/*/*.tab")

GMS_annot = {}
for af in fl:
    with open(af, "r") as infile:
        file_name = af.split("/")[-1].replace(".tab", "")
        GMS_annot.setdefault(file_name, {})
        for line in infile:
            seqnum, gene, pident, length = line.strip("\n").split("\t")[0:4]
            _sp, _gene = gene.split(":")
            GMS_annot[file_name][int(seqnum)] = {"species":_sp, "gene":_gene, "pident":pident, "length":length}


# #### Read MBGD Blast results, assign protein sequences into each clutser

# In[119]:


count, nometa_count = 0, 0
try:
    os.mkdir("MBGD_clusters_faa")
except:
    pass

with open("3-1.CD-HIT/GMS2.faa", "r") as infile:
    for idx, line in enumerate(infile):
        if idx % 100000 == 0:
            print(idx)
            
        if line.startswith(">"):
            skipLine = False
            acc = line.strip("\n").strip(">")
            sample_name = acc.split("|")[0].split(".")[0].strip("_genomic")
            
            if sample_name not in OHtable:
                nometa_count += 1
                skipLine = True
                #print(acc, sample_name) 
                continue
            
            O, H = OHtable[sample_name][:2]            
            
            """
            search_res = meta[meta.acc == sample_name]
        
            if len(search_res) != 1:
                print(acc, sample_name)
                continue
            
            O, H = list(search_res["O"])[0], list(search_res["H"])[0]
            """
            
            file_name, seqnum = acc.split("|")
            if int(seqnum) not in GMS_annot[file_name]:
                count += 1
                skipLine = True
                continue
                
            blast_info = GMS_annot[file_name][int(seqnum)]
            
            if blast_info["species"] not in Gene2Cluster:
                count += 1
                skipLine = True
                continue
                
            if blast_info["gene"] not in Gene2Cluster[blast_info["species"]]:
                count += 1
                skipLine = True
                continue
                                
            ClusterList = [int(key) for key in Gene2Cluster[blast_info["species"]][blast_info["gene"]]]
        
        else:
            if skipLine: continue
                
            for clstr in ClusterList:
                if not os.path.exists("MBGD_clusters_faa/{}".format(clstr%1000)):
                    try:
                        os.mkdir("MBGD_clusters_faa/{}".format(clstr%1000))
                    except:
                        pass
                    
                outfile = open("MBGD_clusters_faa/{}/{}.faa".format(clstr%1000, clstr), "a")
            
                print(">{}:{}:{}:{}:{}".format(acc, O, H, blast_info["pident"], blast_info["length"]), file=outfile)
                print(line, end="", file=outfile)
                
                outfile.close()
        
print(count)
print(nometa_count)


# #### CD-HIT to refine cluster definition
# #### Parse CD-HIT result

# In[26]:


#fl = glob.glob("3-1.CD-HIT/clusters_c90/*.c90n5.clstr")
fl = glob.glob("3-1.CD-HIT/clusters_c50/*.c50n2.clstr")
#fl = glob.glob("3-1.CD-HIT/clusters_c50/117.c50n2.clstr")
print(len(fl))

CDHIT_clstr_dict = {}
for af in fl: 
    MBGD_clusterNo = int(af.split("/")[-1].split(".")[0])
    with open(af, "r") as infile:
        for line in infile:
            if line.startswith(">"):
                CDHIT_clusterNo = int(line.split(" ")[-1].strip("\n"))
            else:
                _seqid = line.split(" ")[1].split(":")[0].strip(">")
                CDHIT_clstr_dict.setdefault(_seqid, [])
                CDHIT_clstr_dict[_seqid].append((MBGD_clusterNo, CDHIT_clusterNo))


# In[27]:


print(len(CDHIT_clstr_dict))


# In[28]:


CDHIT_clstr_dict["ESC_GC3128AA_AS.result|3531"]


# #### Read CD-HIT results, assign gene vectors into each clutser

# In[135]:


count, nometa_count = 0, 0
try:
    os.mkdir("MBGD_clusters_cdhit")
except:
    pass

with open("5-1.ESM/GMS2.faa.txt", "r") as infile:
    for idx, line in enumerate(infile):
        if idx % 100000 == 0:
            print(idx)
            
        if line.startswith(">"):
            skipLine = False
            acc = line.strip("\n").strip(">")
            sample_name = acc.split("|")[0].split(".")[0].strip("_genomic")
      
            if acc not in CDHIT_clstr_dict:
                count += 1
                skipLine = True
                continue
                
            #search_res = meta[meta.acc == sample_name]                
            #if len(search_res) != 1: # Excluded by sample filtering
            if sample_name not in OHtable:
                nometa_count += 1
                skipLine = True
                #print(acc, sample_name) 
                continue
            
            #O, H = list(search_res["O"])[0], list(search_res["H"])[0]
            O, H = OHtable[sample_name][:2]
            
            cdhit_info_list = CDHIT_clstr_dict[acc]
            
            """            
            file_name, seqnum = acc.split("|")
            if int(seqnum) not in GMS_annot[file_name]:
                count += 1
                skipLine = True
                continue
                
            #blast_info = GMS_annot[file_name][int(seqnum)]
            
            
            if blast_info["species"] not in Gene2Cluster:
                count += 1
                skipLine = True
                continue
                
            if blast_info["gene"] not in Gene2Cluster[blast_info["species"]]:
                count += 1
                skipLine = True
                continue
                                
            ClusterList = [int(key) for key in Gene2Cluster[blast_info["species"]][blast_info["gene"]]]
            """
        
        else:
            if skipLine: continue
                
            for cdhit_info in cdhit_info_list:
                if not os.path.exists("MBGD_clusters_cdhit/{}".format(cdhit_info[0]%1000)):
                    try:
                        os.mkdir("MBGD_clusters_cdhit/{}".format(cdhit_info[0]%1000))
                    except:
                        pass

                outfile = open("MBGD_clusters_cdhit/{}/{}_{}.vecs".format(cdhit_info[0]%1000, cdhit_info[0], cdhit_info[1]), "a")

                #print(">{}:{}:{}:{}:{}".format(acc, O, H, blast_info["pident"], blast_info["length"]), file=outfile)
                print(">{}:{}:{}".format(acc, O, H), file=outfile)
                print(line, end="", file=outfile)

                outfile.close()
        
print(count)
print(nometa_count)


# ### 8. Run Random Forest model for each gene cluster

# interpreter version

# In[268]:


for clstr in ClusterMeta:
#for clstr in ["100"]:
    DataMat = {"v{}".format(i):[] for i in range(320)}
    DataMat["answer"] = []
    DataMat["acc"] = []
    temp_vecs = {}
    
    clstr = int(clstr)
    if not os.path.isfile("MBGD_clusters/{}/{}.vecs".format(clstr%1000, clstr)): continue
    
    with open("MBGD_clusters/{}/{}.log4".format(clstr%1000, clstr), "w") as outfile:
        with open("MBGD_clusters/{}/{}.vecs".format(clstr%1000, clstr), "r") as infile:
            for line in infile:
                if line.startswith(">"): # odd lines
                    acc, O, H, pident, length = line.strip("\n").strip(">").split(":")
                    continue
                    
                # even lines; deal with only valid samples
                sample_acc = acc.split("|")[0].split(".")[0].replace("_genomic", "")
                if sample_acc in list(meta["acc"]):
                    vecs = line.strip("\n").split(" ")                    
                    temp_vecs.setdefault(sample_acc, [])
                    temp_vecs[sample_acc].append([float(vec) for vec in vecs])
                    continue
            
            for sample_acc in temp_vecs:
                mean_vec = np.mean(temp_vecs[sample_acc], axis = 0)
                DataMat["acc"].append(sample_acc)
                DataMat["answer"].append(list(meta[meta["acc"] == sample_acc]["O])[0])
                for i in range(len(mean_vec)):
                    DataMat["v{}".format(i)].append(mean_vec[i])
                    
            DataMat = pd.DataFrame(DataMat)

            O_stat = dict(DataMat["answer"].value_counts())
            len(O_stat), sum(O_stat.values())
            
            if sum(O_stat.values()) < 10 or len(O_stat) < 3: continue

            O_most_frequent = max(O_stat.values())/sum(O_stat.values())
            entpy = entropy(list(O_stat.values())/sum(O_stat.values()), base=2)
            normed_entpy = entpy/math.log2(len(O_stat))

            Y = DataMat["answer"].astype("category")
            X = DataMat.drop(["answer", "acc"], axis=1)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

            RF_estimator = RandomForestClassifier(n_estimators=100, random_state=33, n_jobs=6, criterion="entropy", max_depth=50)
            scores = cross_val_score(RF_estimator, X, Y, cv=StratifiedKFold(n_splits = 10))

            print("File : {}, Accuracy Mean : {}, Accuracy Std : {}, Entropy(Normalized) : {}, Num of O classes : {}, Num of Data : {} (Excluded {}), Frequency of Most Prevalent O type : {}".format(clstr, scores.mean(), scores.std(), normed_entpy, len(O_stat), sum(O_stat.values()), len(meta)-sum(O_stat.values()), O_most_frequent),
                 file=outfile)


# compiler version (for multiprocessing)

# In[ ]:


cmd_list = ["python3 sub.py {}".format(clstr) for clstr in ClusterMeta]
def worker(cmd): 
    p = Popen(cmd, shell=True)
    p.wait()

pool = Pool( processes = 40 )
results =[pool.apply_async(worker, [cmd]) for cmd in cmd_list]
ans = [res.get() for res in results]


# ### 9. Screen Genes related to O serotype determination

# In[38]:


get_ipython().system('cat MBGD_clusters_cdhit_c90/*/*.log5 > MBGD_genes.log5')


# In[39]:


criterion_count = 0
acc_values = []
genome_values = []
Otype_values = []

target_clstr = set()
print("\t".join(["numO", "numData", "accuracy_mean", "accuracy_std", "major O", "entpy", "MBGD_clstrID", "CDHIT_clstrID", "gene_name", "gene_description"]))

for af in glob.glob("MBGD_clusters_cdhit/*/*.log5"):
    #with open("MBGD_genes.log5", "r") as infile:
    with open(af, "r") as infile:
        for line in infile:
            clstr, accuracy_mean, accuracy_std, norm_entpy, numO, numData, numMajor = [val.split(" : ")[-1] for val in line.strip("\n").split(",")]
            clstrMBGD, clstrCDHIT, numO = int(clstr.split("_")[0]), clstr.split("_")[1], int(numO)
            numData, numData_excluded = [int(a) for a in numData.rstrip(")").split(" (Excluded ")]
            genome_values.append(numData)
            Otype_values.append(numO)
            accuracy_mean, accuracy_std, norm_entpy, numMajor = float(accuracy_mean), float(accuracy_std), float(norm_entpy), float(numMajor)
            
            if numO >=10 and int(numData) > 1000:
                acc_values.append(accuracy_mean)

            if numO >=10 and numData >= 5000 and numMajor < 0.4 and accuracy_mean >= 0.75:
                if numO >=10 and numData >= 8000 and numMajor < 0.4 and accuracy_mean >= 0.75:
                    print("*{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}".format(numO, numData, accuracy_mean, accuracy_std, numMajor, norm_entpy, clstrMBGD, clstrCDHIT, ClusterMeta[str(clstrMBGD)]["gene"], ClusterMeta[str(clstrMBGD)]["descr"]), sep="\t")

                    target_clstr.add((clstrMBGD, clstrCDHIT))
                    criterion_count += 1
                else:
                    print("{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{}\t{}\t{}\t{}".format(numO, numData, accuracy_mean, accuracy_std, numMajor, norm_entpy, clstrMBGD, clstrCDHIT, ClusterMeta[str(clstrMBGD)]["gene"], ClusterMeta[str(clstrMBGD)]["descr"]), sep="\t")

    #print(criterion_count)


# In[40]:


# Duplicated WcaL records
target_clstr.remove((1838, "0"))
target_clstr.remove((2656, "0"))


# In[185]:


acc_values = np.array(acc_values, dtype='float')


# In[225]:


rcParams['figure.figsize'] = 4,1.5
plt.style.use('ggplot')
plt.hist(acc_values, bins=30, color='tab:cyan', edgecolor="white")
plt.xticks(fontsize=6)#, rotation=90)
plt.yticks(fontsize=6)
#plt.yscale('log')
plt.xlim(0,1)
plt.ylabel('# of Genes', fontsize=6)
plt.xlabel('prediction accuracy', fontsize=6)
#plt.show()
plt.margins(0)
plt.savefig("fig2c.png")


# In[1679]:


genome_values = np.array(genome_values, dtype='float')
rcParams['figure.figsize'] = 2,1.5
plt.style.use('ggplot')
plt.hist(genome_values, bins=30, color='tab:blue', edgecolor="white")
plt.xticks(fontsize=6)#, rotation=90)
plt.yticks(fontsize=6)
#plt.yscale('log')
#plt.xlim(0,1)
plt.ylabel('Genes', fontsize=6)
plt.xlabel('# of Genomes', fontsize=6)
#plt.show()
plt.margins(0)
plt.savefig("fig2c.png")


# In[1689]:


len(genome_values[genome_values>8000])
#len(genome_values)


# In[1681]:


Otype_values = np.array(Otype_values, dtype='float')
rcParams['figure.figsize'] = 2,1.5
plt.style.use('ggplot')
plt.hist(Otype_values, bins=30, color='tab:green', edgecolor="white")
plt.xticks(fontsize=6)#, rotation=90)
plt.yticks(fontsize=6)
#plt.yscale('log')
#plt.xlim(0,1)
plt.ylabel('Genes', fontsize=6)
plt.xlabel('# of Genomes', fontsize=6)
#plt.show()
plt.margins(0)
plt.savefig("fig2c.png")


# In[1692]:


len(Otype_values[Otype_values<150])


# ### 10. Compare Classfication Performance for each genes

# In[496]:


gene_clsf_list = []
dimVec = 320
temp_vecs = {}

for clstrMBGD, clstrCDHIT in target_clstr:
    DataMat = {"acc":[], "answer":[]}
    for i in range(dimVec):
        DataMat["MBGD{}_CDHIT{}_v{}".format(clstrMBGD, clstrCDHIT, i)] = []        
        
    with open("MBGD_clusters_cdhit/{}/{}_{}.vecs".format(int(clstrMBGD)%1000, clstrMBGD, clstrCDHIT), "r") as infile:
        for line in infile:
            if line.startswith(">"): # odd lines
                acc, O, H = line.strip("\n").strip(">").split(":")[:3]
                continue
            else:
                sample_acc = acc.split("|")[0].split(".")[0].replace("_genomic", "")
                if sample_acc in OHtable:
                    vecs = line.strip("\n").split(" ")
                    temp_vecs.setdefault(sample_acc, [])
                    temp_vecs[sample_acc].append([float(vec) for vec in vecs])
                    
        duplicate_count = 0
        for sample_acc in temp_vecs:
            mean_vec = np.mean(temp_vecs[sample_acc], axis = 0)
            DataMat["acc"].append(sample_acc)
            DataMat["answer"].append(OHtable[sample_acc][0])
            for i in range(len(mean_vec)):
                DataMat["MBGD{}_CDHIT{}_v{}".format(clstrMBGD, clstrCDHIT, i)].append(mean_vec[i])
            if len(temp_vecs[sample_acc]) > 1: duplicate_count += 1
        #print(duplicate_count)
        
    gene_clsf_list.append({"DataMat":pd.DataFrame(DataMat),
                           "clstrID":{"MBGD":clstrMBGD, "CDHIT":clstrCDHIT},
                          })


# In[523]:


gene_clsf_list[0]["DataMat"]["answer"].value_counts()[:20]


# In[552]:


for gene_clsf in gene_clsf_list:
    nFold = 5.0
    target_O_list = [label for label, count in dict(gene_clsf["DataMat"]["answer"].value_counts()).items() if count >= nFold]
    DataMat_test = gene_clsf["DataMat"]
    DataMat_test = DataMat_test[DataMat_test["answer"].isin(target_O_list)]
    
    Y = DataMat_test["answer"].astype("category")
    X = DataMat_test.drop(["answer", "acc"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/nFold, stratify=Y)

    RF_estimator = RandomForestClassifier(n_estimators=100, random_state=int(time.time()), n_jobs=40, criterion="entropy", max_depth=10,
                                         class_weight="balanced")
    RF_estimator.fit(X, Y)
    
    major_labels = list(Y.value_counts()[:10].keys())
    minor_labels = list(Y.value_counts()[-100:-90].keys())
    y_pred = RF_estimator.predict(x_test)    
    
    gene_clsf["clf"] = RF_estimator
    gene_clsf["major_labels"] = major_labels
    gene_clsf["minor_labels"] = minor_labels
    gene_clsf["conf_matrix_major"] = metrics.confusion_matrix(y_test, y_pred, labels=major_labels)
    gene_clsf["conf_matrix_minor"] = metrics.confusion_matrix(y_test, y_pred, labels=minor_labels)
    #gene_clsf["clf_report"] = classification_report(y_test, y_pred)
    gene_clsf["clf_report"] = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    #ene_clsf["clf_report_major"] = classification_report(y_test, y_pred, labels=major_labels)
    #ene_clsf["clf_report_minor"] = classification_report(y_test, y_pred, labels=minor_labels)


# In[1398]:


major_O_list = list(dict(meta["O"].value_counts()).keys())[0:5]
#major_O_list.remove("O111")
middle_O_list = list(dict(meta["O"].value_counts()).keys())[100:105]

nFold = 5
skf = StratifiedKFold(n_splits=nFold)
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/nFold, stratify=Y)

for gene_clsf in gene_clsf_list:    
    target_O_list = [label for label, count in dict(gene_clsf["DataMat"]["answer"].value_counts()).items() if count >= nFold]
    DataMat_test = gene_clsf["DataMat"]
    DataMat_test = DataMat_test[DataMat_test["answer"].isin(target_O_list)]
    
    Y = DataMat_test["answer"].astype("category")
    X = DataMat_test.drop(["answer", "acc"], axis=1)
    
    #all_conf_matrix_major = np.zeros((len(Y.cat.categories), len(Y.cat.categories)))
    #all_conf_matrix_minor = np.zeros((len(Y.cat.categories), len(Y.cat.categories)))
    all_conf_matrix_major = np.zeros((len(major_O_list), len(major_O_list)))
    all_conf_matrix_minor = np.zeros((len(middle_O_list), len(middle_O_list)))
    clf_reports = []    
    
    for train_index, test_index in skf.split(X, Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        RF_estimator = RandomForestClassifier(n_estimators=100, random_state=int(time.time()), n_jobs=40, criterion="entropy", max_depth=10,
                                              class_weight="balanced")
        RF_estimator.fit(x_train, y_train)
        y_pred = RF_estimator.predict(x_test)
        
        #major_labels = list(Y.value_counts()[:10].keys())
        #minor_labels = list(Y.value_counts()[-100:-90].keys())
        
        all_conf_matrix_major += confusion_matrix(y_test, y_pred, labels=major_O_list)
        all_conf_matrix_minor += confusion_matrix(y_test, y_pred, labels=middle_O_list)
        clf_reports.append(classification_report(y_test, y_pred, output_dict=True))
    
    gene_clsf["clf"] = RF_estimator
    gene_clsf["conf_matrix_major"] = all_conf_matrix_major / nFold
    gene_clsf["conf_matrix_minor"] = all_conf_matrix_minor / nFold
    gene_clsf["clf_report_mean"] = clf_report_mean
    gene_clsf["clf_report_std"] = clf_report_std


# In[1399]:


all_conf_matrix_major


# In[553]:


OtypeList = ["O157", "O26", "O103", "O8", "O121", "O111", "O25", "O149", "O145", "O6"]


# In[554]:


print(ClusterMeta[str(gene_clsf_list[0]["clstrID"]["MBGD"])]["gene"])
# show major results
display(gene_clsf_list[0]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[[i for i in OtypeList if i != "O111"]])
#display(gene_clsf_list[0]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[["O157"]])
display(gene_clsf_list[0]["clf_report"].loc[["accuracy", "macro avg", "weighted avg"]])


# In[1397]:


ConfusionMatrixDisplay(gene_clsf_list[0]["conf_matrix_major"], display_labels=gene_clsf_list[0]["major_labels"]).plot()


# In[136]:


ConfusionMatrixDisplay(gene_clsf_list[0]["conf_matrix_minor"], display_labels=gene_clsf_list[0]["minor_labels"]).plot()


# In[556]:


print(ClusterMeta[str(gene_clsf_list[1]["clstrID"]["MBGD"])]["gene"])
# show major results
display(gene_clsf_list[1]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[OtypeList])
#display(gene_clsf_list[0]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[["O157"]])
display(gene_clsf_list[1]["clf_report"].loc[["accuracy", "macro avg", "weighted avg"]])


# In[557]:


print(ClusterMeta[str(gene_clsf_list[2]["clstrID"]["MBGD"])]["gene"])
# show major results
display(gene_clsf_list[2]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[OtypeList])
#display(gene_clsf_list[0]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[["O157"]])
display(gene_clsf_list[2]["clf_report"].loc[["accuracy", "macro avg", "weighted avg"]])


# In[558]:


print(ClusterMeta[str(gene_clsf_list[3]["clstrID"]["MBGD"])]["gene"])
# show major results
display(gene_clsf_list[3]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[OtypeList])
#display(gene_clsf_list[0]["clf_report"].drop(["accuracy", "macro avg", "weighted avg"]).sort_values("support", ascending=False).loc[["O157"]])
display(gene_clsf_list[3]["clf_report"].loc[["accuracy", "macro avg", "weighted avg"]])


# In[573]:


OtypeList = ["O157", "O26", "O103", "O8", "O121", "O25", "O149", "O145", "O6"] # O111
prc_table = gene_clsf_list[0]["clf_report"].loc[OtypeList][["precision"]].rename(
    columns={"precision":ClusterMeta[str(gene_clsf_list[0]["clstrID"]["MBGD"])]["gene"]})

for i in range(1, len(gene_clsf_list)):
    prc_table = pd.merge(prc_table,
                         gene_clsf_list[i]["clf_report"].loc[OtypeList][["precision"]].rename(
                           columns={"precision":ClusterMeta[str(gene_clsf_list[i]["clstrID"]["MBGD"])]["gene"]}),
                         left_index=True, right_index=True, how="outer")


# In[574]:


prc_table


# In[1484]:


RF_estimator = RandomForestClassifier(n_estimators=100, random_state=int(time.time()), n_jobs=40, criterion="entropy", max_depth=10,
                                              class_weight="balanced")

le = LabelEncoder()  

precision = cross_val_score(RF_estimator,
                            gene_clsf_list[0]["DataMat"].drop(["answer", "acc"], axis=1),
                            gene_clsf_list[0]["DataMat"]["answer"].astype("category"),
                            cv=10, scoring='precision_micro')
#le.fit_transform(


# In[1614]:


for i in range(len(gene_clsf_list)):
    gene_clsf_list[i]["CVPredict"] = cross_val_predict(RF_estimator,
                  gene_clsf_list[i]["DataMat"].drop(["answer", "acc"], axis=1),
                  gene_clsf_list[i]["DataMat"]["answer"].astype("category"),
                  cv=StratifiedKFold(n_splits = 10))


# In[1626]:


pd.DataFrame(classification_report(gene_clsf_list[0]["DataMat"]["answer"],
                                   gene_clsf_list[0]["CVPredict"],
                                   output_dict=True)).T.drop(["accuracy", "macro avg", "weighted avg"]).sort_values(
                                    "support", ascending=False).loc[OtypeList]


# In[1710]:


OtypeList = ["O157", "O26", "O103", "O8", "O121", "O111", "O25", "O149", "O145", "O6"] # O111
i = 0
def prc_temp(i):
    prc_temp = pd.DataFrame(classification_report(gene_clsf_list[i]["DataMat"]["answer"],
                                   gene_clsf_list[i]["CVPredict"],
                                   output_dict=True)).T.drop(["accuracy", "macro avg", "weighted avg"]).sort_values(
                                    "support", ascending=False).loc[OtypeList]
    return prc_temp

prc_table = prc_temp(0)[["precision"]].rename(
    columns={"precision":ClusterMeta[str(gene_clsf_list[0]["clstrID"]["MBGD"])]["gene"]})

for i in range(1, len(gene_clsf_list)):    
    prc_table = pd.merge(prc_table,
                         prc_temp(i)[["precision"]].rename(
                           columns={"precision":ClusterMeta[str(gene_clsf_list[i]["clstrID"]["MBGD"])]["gene"]}),
                         left_index=True, right_index=True, how="outer")


# In[1711]:


prc_table


# In[1632]:


df_melted = prc_table.T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
df_melted.rename(columns={'index': 'Type'}, inplace=True)

#pastel_palette = sns.color_palette("deep")
# 박스 플롯 생성
plt.figure(figsize=(4, 5))
sns.set(style="white")
barplot = sns.barplot(x='Value', y='Feature', hue='Type', data=df_melted, alpha=1, palette=sns.color_palette("pastel"))
barplot.legend_.remove()
pointplot = sns.pointplot(x='Value', y='Feature', hue='Type', data=df_melted, dodge=0.75, join=False, palette=sns.color_palette("deep"), markers='o', scale=0.4, ci=None)

plt.title('Precision by O serotype', fontsize=10)
plt.xlabel('Precision', fontsize=8)
plt.ylabel('')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
#plt.legend(title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#.yaxis.set_major_locator(MultipleLocator(0.2))
plt.grid(True, axis="x", alpha=0.4)

handles, labels = pointplot.get_legend_handles_labels()
plt.legend(handles=handles[0:len(df_melted['Type'].unique())], labels=labels[0:len(df_melted['Type'].unique())], title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.show()


# In[1682]:


#df_melted = prc_table.loc[["O157", "O26", "O103", "O8", "O121"]].T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
df_melted = prc_table.T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
df_melted.rename(columns={'index': 'Type'}, inplace=True)

# 박스 플롯 생성
plt.figure(figsize=(8, 4))  # 가로 방향으로 그릴 때, 가로 폭을 더 크게 설정
sns.set(style="white")
barplot = sns.barplot(x='Feature', y='Value', hue='Type', data=df_melted, alpha=1, palette=sns.color_palette("pastel"))
barplot.legend_.remove()
pointplot = sns.pointplot(x='Feature', y='Value', hue='Type', data=df_melted, dodge=0.75, join=False, palette=sns.color_palette("deep"), markers='o', scale=0.4, ci=None)

plt.title('Precision by O serotype', fontsize=10)
plt.ylabel('Precision', fontsize=8)  # y 축 이름을 Precision으로 변경
plt.xlabel('')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, axis="y", alpha=0.4)  # y 축에 그리드 추가

# 범례 설정
handles, labels = pointplot.get_legend_handles_labels()
plt.legend(handles=handles[0:len(df_melted['Type'].unique())], labels=labels[0:len(df_melted['Type'].unique())], title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.show()


# In[1120]:


df_melted = prc_table.T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
df_melted.rename(columns={'index': 'Type'}, inplace=True)

#pastel_palette = sns.color_palette("deep")
# 박스 플롯 생성
plt.figure(figsize=(4, 5))
sns.set(style="white")
barplot = sns.barplot(x='Value', y='Feature', hue='Type', data=df_melted, alpha=1, palette=sns.color_palette("pastel"))
barplot.legend_.remove()
pointplot = sns.pointplot(x='Value', y='Feature', hue='Type', data=df_melted, dodge=0.75, join=False, palette=sns.color_palette("deep"), markers='o', scale=0.4, ci=None)

plt.title('Precision by O serotype', fontsize=10)
plt.xlabel('Precision', fontsize=8)
plt.ylabel('')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
#plt.legend(title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#.yaxis.set_major_locator(MultipleLocator(0.2))
plt.grid(True, axis="x", alpha=0.4)

handles, labels = pointplot.get_legend_handles_labels()
plt.legend(handles=handles[0:len(df_melted['Type'].unique())], labels=labels[0:len(df_melted['Type'].unique())], title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.show()


# In[1662]:


OtypeList = ["O157", "O26", "O103", "O8", "O121", "O25", "O149", "O145", "O6"] # O111
i = 0
def recall_temp(i):
    recall_temp = pd.DataFrame(classification_report(gene_clsf_list[i]["DataMat"]["answer"],
                                   gene_clsf_list[i]["CVPredict"],
                                   output_dict=True)).T.drop(["accuracy", "macro avg", "weighted avg"]).sort_values(
                                    "support", ascending=False).loc[OtypeList]
    return recall_temp

recall_table = recall_temp(0)[["recall"]].rename(
    columns={"recall":ClusterMeta[str(gene_clsf_list[0]["clstrID"]["MBGD"])]["gene"]})

for i in range(1, len(gene_clsf_list)):    
    recall_table = pd.merge(recall_table,
                         recall_temp(i)[["recall"]].rename(
                           columns={"recall":ClusterMeta[str(gene_clsf_list[i]["clstrID"]["MBGD"])]["gene"]}),
                         left_index=True, right_index=True, how="outer")


# In[1663]:


recall_table


# In[1664]:


df_melted = recall_table.T.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
df_melted.rename(columns={'index': 'Type'}, inplace=True)

#pastel_palette = sns.color_palette("deep")
# 박스 플롯 생성
plt.figure(figsize=(4, 5))
sns.set(style="white")
barplot = sns.barplot(x='Value', y='Feature', hue='Type', data=df_melted, alpha=1, palette=sns.color_palette("pastel"))
barplot.legend_.remove()
pointplot = sns.pointplot(x='Value', y='Feature', hue='Type', data=df_melted, dodge=0.75, join=False, palette=sns.color_palette("deep"), markers='o', scale=0.4, ci=None)

plt.title('Recall by O serotype', fontsize=10)
plt.xlabel('Recall', fontsize=8)
plt.ylabel('')
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
#plt.legend(title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
#.yaxis.set_major_locator(MultipleLocator(0.2))
plt.grid(True, axis="x", alpha=0.4)

handles, labels = pointplot.get_legend_handles_labels()
plt.legend(handles=handles[0:len(df_melted['Type'].unique())], labels=labels[0:len(df_melted['Type'].unique())], title='Genes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.show()


# ### 10. MSA analysisfor each gene clusters

# In[31]:


for clstrMBGD, clstrCDHIT in target_clstr:
    with open("MSA/{}_{}.faa".format(clstrMBGD, clstrCDHIT), "w") as outfile: 
        with open("MBGD_clusters_faa/{}/{}.faa".format(clstrMBGD%1000, clstrMBGD)) as infile:
            for record in SeqIO.parse(infile, "fasta"):            
                for comp in CDHIT_clstr_dict[record.id.split(":")[0]]:
                    if comp == (clstrMBGD, int(clstrCDHIT)):
                        SeqIO.write(record, outfile, "fasta")


# In[235]:


def alnSiteCompositionDF(aln, characters="ACDEFGHIKLMNPQRSTVWY"):
    alnRows = aln.get_alignment_length()
    compDict = {char:[0]*alnRows for char in characters}
    for record in aln:
        header = record.id
        seq = record.seq
        for aaPos in range(len(seq)):
            aa = seq[aaPos]
            if aa in characters:
                compDict[aa][aaPos] += 1    
    return pd.DataFrame.from_dict(compDict)


# #### First, check with HisD gene 

# In[892]:


target_file = "684_0" #684_0, 117_3, 66_0

am = AlignIO.read("MSA/{}.addfrag.mafft.msa".format(target_file), "fasta")

###

first_seq = am[0]
positions_to_remove = [pos for pos, char in enumerate(first_seq) if char == '-']
new_alignment = MultipleSeqAlignment([])
for record in am:
    new_seq = "".join([char for pos, char in enumerate(record.seq) if pos not in positions_to_remove])
    new_record = SeqRecord(Seq(new_seq), id=record.id, description=record.description)
    new_alignment.append(new_record)
    
am = new_alignment

###

alignment_length = am.get_alignment_length()
aa_composition = []

for i in range(alignment_length):
    column = am[:, i]
    aa_counts = {}
    total_count = len(column)

    for aa in column:
        if aa not in aa_counts:
            aa_counts[aa] = 1
        else:
            aa_counts[aa] += 1

    # Calculate the frequency of each amino acid
    aa_frequencies = {aa: count / total_count for aa, count in aa_counts.items()}
    aa_composition.append(aa_frequencies)
    
###

max_composition = [max(position.values()) for position in aa_composition]
df_max_composition = pd.DataFrame({
    'Position': range(1, len(max_composition) + 1),
    'Max Composition': max_composition
})


# In[1087]:


target_file = "684_0" 

plt.figure(figsize=(16, 9))
target_pos = 48
OtypeList = ["O157", "O26", "O103", "O8", "O121", "O111", "O25", "O149", "O145", "O9"]

### 
### 제일 윗부분 LOGO 그리는 부분 ###
### + 위치별 통계
###

msa_group_by = {}
target_pos_ratio = {}
for record in am:
    Otype = record.id.split(":")[1]
    msa_group_by.setdefault(Otype, [])
    msa_group_by[Otype].append((str(record.seq[40:50]), str(record.seq[272:282])))
    
    target_pos_ratio.setdefault(record.seq[target_pos], {})
    target_pos_ratio[record.seq[target_pos]].setdefault(Otype, 0)
    target_pos_ratio[record.seq[target_pos]][Otype] += 1

target_res = sorted(target_pos_ratio.items(), key=lambda item: sum(item[1].values()), reverse=True)
#print(target_res[0])
    
###

outermost_grid = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.2)

###



right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], subplot_spec=outermost_grid[1])


rx1 = plt.subplot(right_grid[0])
first_res = target_res[0]
O_stats = sorted(first_res[1].items(), key=lambda item: item[1], reverse=True)
#print(O_stats)

def my_autopct_1(pct):
    return ('%1.1f%%' % pct) if pct >= 1.5 else ''

rx1.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 6 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels
        shadow=True,
        pctdistance=0.80,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_1,
        explode=[0.1]*len(O_stats),
        startangle=90,
       )

rx2 = plt.subplot(right_grid[1])
second_res = target_res[1]

O_stats = sorted(second_res[1].items(), key=lambda item: item[1], reverse=True)

def my_autopct_2(pct):
    return ('%1.1f%%' % pct) if pct >= 3.0 else ''

rx2.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 7 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels 
        shadow=True,
        pctdistance=0.8,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_2,
        explode=[0.2]*len(O_stats),
        startangle=90,
       )


###

outer_grid = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[6, 1, 1.5], subplot_spec=outermost_grid[0])#, hspace=0.2)

   

#plt.style.use('default')

inner_grid = gridspec.GridSpecFromSubplotSpec(nrows=len(OtypeList), ncols=2, subplot_spec=outer_grid[0], width_ratios=[1, 1])

#axs1, axs2, axs3 = plt.subplot(inner_grid[0]), plt.subplot(inner_grid[1]), plt.subplot(inner_grid[2])
#fig, axs = ax1.subplots(nrows=len(OtypeList), ncols=2, figsize=[7,4], gridspec_kw={'width_ratios': [1, 1]})
highlights = [[1,4,8],
              [1,2,6,7],
             ]

for idx, Otype in enumerate(OtypeList):
    for i in range(2):
        #ax = fig.add_subplot()
        #ax = axs[idx,i]
        #ax = fig.add_subplot(gs[idx, i])
        ax = plt.subplot(inner_grid[idx, i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if i == 0:
            ax.set_ylabel(OtypeList[idx], rotation="horizontal")
            ax.yaxis.set_label_coords(-0.1, 0.2)
        target_alignment = [seqs[i] for seqs in  msa_group_by[Otype]]
        #print(msa_group_by[Otype])
        counts_mat = lm.alignment_to_matrix(target_alignment)
        aa_logo = lm.Logo(counts_mat, color_scheme="charge", ax=ax, alpha=0.5)
        #aa_logo.style_xticks(anchor=0, spacing=5)
        for position in highlights[i]:
            aa_logo.highlight_position(p=position, color="Yellow", alpha=0.95)

###
### InterProScan 결과 그리는 부분 ###
###

features = []

with open("MSA/{}.interproscan.tsv".format(target_file), "r") as infile:
    for line in infile:
        fields = line.strip().split("\t")
        feat_name = fields[3]+" - "+fields[5]
        #feat_name = fields[3]
        #print(feat_name)
        if fields[3] not in ["ProSitePatterns"]: continue #Pfam
        _start, _end = int(fields[6]), int(fields[7])
        
        features.append(GraphicFeature(start=_start, end=_end, strand=0,
                                       label = feat_name,
                                       #label=None,
                                       open_left=True,
                                       open_right=True,
                                       color="#dbd9b6",
                                       linecolor="#dbd9b6",
        ))
        
with open("MSA/684.uniprot.features.json", "r") as infile:
    uniprot_features = json.load(infile)
    
for uniprot_feature in uniprot_features["features"]:
    _start = uniprot_feature["location"]["start"]["value"]
    _end = uniprot_feature["location"]["end"]["value"]
    #_name = ""
    flag = False
    
    if "ligand" in uniprot_feature:
        _name = None
        
        #_name = uniprot_feature["ligand"]["name"] + uniprot_feature["type"]
        #_name = uniprot_feature["ligand"]["name"]
        
        if uniprot_feature["ligand"]["name"] == "NAD(+)": _color, _linecolor, _alpha = "red", "red", 1
        elif uniprot_feature["ligand"]["name"] == "Zn(2+)": _color, _linecolor, _alpha = "#0c17e8", "#0c17e8", 0.4
        elif uniprot_feature["ligand"]["name"] == "substrate": _color, _linecolor, _alpha = "#0abf49", "#0abf49", 1
        
        #print(_start, _end, _color, _linecolor)
        
        flag = True
        
    elif uniprot_feature["type"] == "Active site":
        
        #_name = uniprot_feature["description"] + uniprot_feature["type"]
        #_name = uniprot_feature["description"]
        
        _name = None
        _color, _linecolor, _alpha = "skyblue", "skyblue", 1
        flag = True
        
    if flag:
        features.append(GraphicFeature(start=_start, end=_end, strand=0,
                                       label = _name,
                                       #label = None,
                                       open_left=True,
                                       open_right=True,
                                       color=_color,
                                       linecolor = _linecolor,
                                       alpha=_alpha, # 의미 없음
        ))
    

ax2 = plt.subplot(outer_grid[2])
record = GraphicRecord(sequence_length=max(df_max_composition['Position']), features=features, feature_level_height=0)
#record._format_label(feat_name, max_label_length=200, max_line_length=200)
record.plot(ax=ax2, max_label_length=100, max_line_length=40)#, figure_width=10)
#record.plot(ax=ax2)

ax3 = plt.subplot(outer_grid[1])
ax3.plot(df_max_composition['Position'], df_max_composition['Max Composition'], marker='', color="black", alpha=0.9, linewidth=1)
#ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.margins(x=0)
ax3.set_ylim(0.5,1)


# In[1061]:


feature_json["features"][0]["type"] == "Active site"


# In[1037]:


feature_json["features"][10]["type"]
feature_json["features"][10]["ligand"]["name"]
feature_json["features"][10]


# #### garR gene

# In[882]:


target_file = "66_0" #684_0, 117_3, 66_0

am = AlignIO.read("MSA/{}.addfrag.mafft.msa".format(target_file), "fasta")

###

first_seq = am[0]
positions_to_remove = [pos for pos, char in enumerate(first_seq) if char == '-']
new_alignment = MultipleSeqAlignment([])
for record in am:
    new_seq = "".join([char for pos, char in enumerate(record.seq) if pos not in positions_to_remove])
    new_record = SeqRecord(Seq(new_seq), id=record.id, description=record.description)
    new_alignment.append(new_record)
    
am = new_alignment

###

alignment_length = am.get_alignment_length()
aa_composition = []

for i in range(alignment_length):
    column = am[:, i]
    aa_counts = {}
    total_count = len(column)

    for aa in column:
        if aa not in aa_counts:
            aa_counts[aa] = 1
        else:
            aa_counts[aa] += 1

    # Calculate the frequency of each amino acid
    aa_frequencies = {aa: count / total_count for aa, count in aa_counts.items()}
    aa_composition.append(aa_frequencies)
    
###

max_composition = [max(position.values()) for position in aa_composition]
df_max_composition = pd.DataFrame({
    'Position': range(1, len(max_composition) + 1),
    'Max Composition': max_composition
})


# In[886]:


target_file = "66_0" 

plt.figure(figsize=(16, 9))
target_pos = 48
OtypeList = ["O157", "O26", "O103", "O8", "O121", "O111", "O25", "O149", "O145", "O9"]

### 
### 제일 윗부분 LOGO 그리는 부분 ###
### + 위치별 통계
###

msa_group_by = {}
target_pos_ratio = {}
for record in am:
    Otype = record.id.split(":")[1]
    msa_group_by.setdefault(Otype, [])
    msa_group_by[Otype].append((str(record.seq[40:50]), str(record.seq[270:280])))
    
    target_pos_ratio.setdefault(record.seq[target_pos], {})
    target_pos_ratio[record.seq[target_pos]].setdefault(Otype, 0)
    target_pos_ratio[record.seq[target_pos]][Otype] += 1

target_res = sorted(target_pos_ratio.items(), key=lambda item: sum(item[1].values()), reverse=True)
#print(target_res[0])
    
###

outermost_grid = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.2)

###



right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], subplot_spec=outermost_grid[1])


rx1 = plt.subplot(right_grid[0])
first_res = target_res[0]
O_stats = sorted(first_res[1].items(), key=lambda item: item[1], reverse=True)
#print(O_stats)

def my_autopct_1(pct):
    return ('%1.1f%%' % pct) if pct >= 1.5 else ''

rx1.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 6 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels
        shadow=True,
        pctdistance=0.80,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_1,
        explode=[0.1]*len(O_stats),
        startangle=90,
       )

rx2 = plt.subplot(right_grid[1])
second_res = target_res[1]

O_stats = sorted(second_res[1].items(), key=lambda item: item[1], reverse=True)

def my_autopct_2(pct):
    return ('%1.1f%%' % pct) if pct >= 3.0 else ''

rx2.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 7 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels 
        shadow=True,
        pctdistance=0.8,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_2,
        explode=[0.2]*len(O_stats),
        startangle=90,
       )


###

outer_grid = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[6, 1, 1.5], subplot_spec=outermost_grid[0], hspace=0.2)

   

#plt.style.use('default')

inner_grid = gridspec.GridSpecFromSubplotSpec(nrows=len(OtypeList), ncols=2, subplot_spec=outer_grid[0], width_ratios=[1, 1])

#axs1, axs2, axs3 = plt.subplot(inner_grid[0]), plt.subplot(inner_grid[1]), plt.subplot(inner_grid[2])
#fig, axs = ax1.subplots(nrows=len(OtypeList), ncols=2, figsize=[7,4], gridspec_kw={'width_ratios': [1, 1]})
highlights = [[1,4,8],
              [3,4,8,9],
             ]

for idx, Otype in enumerate(OtypeList):
    for i in range(2):
        #ax = fig.add_subplot()
        #ax = axs[idx,i]
        #ax = fig.add_subplot(gs[idx, i])
        ax = plt.subplot(inner_grid[idx, i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if i == 0:
            ax.set_ylabel(OtypeList[idx], rotation="horizontal")
            ax.yaxis.set_label_coords(-0.1, 0.2)
        target_alignment = [seqs[i] for seqs in  msa_group_by[Otype]]
        #print(msa_group_by[Otype])
        counts_mat = lm.alignment_to_matrix(target_alignment)
        aa_logo = lm.Logo(counts_mat, color_scheme="charge", ax=ax)
        #aa_logo.style_xticks(anchor=0, spacing=5)
        for position in highlights[i]:
            aa_logo.highlight_position(p=position, color="gold", alpha=.7)

###
### InterProScan 결과 그리는 부분 ###
###

features = []

with open("MSA/{}.interproscan.tsv".format(target_file), "r") as infile:
    for line in infile:
        fields = line.strip().split("\t")
        feat_name = fields[3]+" - "+fields[5]
        #print(feat_name)
        #if fields[3] not in ["ProSitePatterns", "Pfam"]: continue #Pfam
        if "bind" not in feat_name:
            if fields[3] not in ["ProSitePatterns"]:
                continue
        else:
            if fields[3] in ["SUPERFAMILY"]:
                continue
        _start, _end = int(fields[6]), int(fields[7])
        
        features.append(GraphicFeature(start=_start, end=_end, strand=0,
                                       label = feat_name
        ))

ax2 = plt.subplot(outer_grid[2])
record = GraphicRecord(sequence_length=max(df_max_composition['Position']), features=features)
record.plot(ax=ax2)#, figure_width=10)

ax3 = plt.subplot(outer_grid[1])
ax3.plot(df_max_composition['Position'], df_max_composition['Max Composition'], marker='', color="black", alpha=0.9, linewidth=1)
#ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.margins(x=0)
ax3.set_ylim(0.5,1)


# #### check with GlmM gene 

# In[887]:


target_file = "117_3" #684_0, 117_3, 66_0

am = AlignIO.read("MSA/{}.addfrag.mafft.msa".format(target_file), "fasta")

###

first_seq = am[0]
positions_to_remove = [pos for pos, char in enumerate(first_seq) if char == '-']
new_alignment = MultipleSeqAlignment([])
for record in am:
    new_seq = "".join([char for pos, char in enumerate(record.seq) if pos not in positions_to_remove])
    new_record = SeqRecord(Seq(new_seq), id=record.id, description=record.description)
    new_alignment.append(new_record)
    
am = new_alignment

###

alignment_length = am.get_alignment_length()
aa_composition = []

for i in range(alignment_length):
    column = am[:, i]
    aa_counts = {}
    total_count = len(column)

    for aa in column:
        if aa not in aa_counts:
            aa_counts[aa] = 1
        else:
            aa_counts[aa] += 1

    # Calculate the frequency of each amino acid
    aa_frequencies = {aa: count / total_count for aa, count in aa_counts.items()}
    aa_composition.append(aa_frequencies)
    
###

max_composition = [max(position.values()) for position in aa_composition]
df_max_composition = pd.DataFrame({
    'Position': range(1, len(max_composition) + 1),
    'Max Composition': max_composition
})


# In[889]:


target_file = "117_3" 

plt.figure(figsize=(16, 9))
target_pos = 48
OtypeList = ["O157", "O26", "O103", "O8", "O121", "O111", "O25", "O149", "O145", "O9"]

### 
### 제일 윗부분 LOGO 그리는 부분 ###
### + 위치별 통계
###

msa_group_by = {}
target_pos_ratio = {}
for record in am:
    Otype = record.id.split(":")[1]
    msa_group_by.setdefault(Otype, [])
    msa_group_by[Otype].append((str(record.seq[40:50]), str(record.seq[270:280])))
    
    target_pos_ratio.setdefault(record.seq[target_pos], {})
    target_pos_ratio[record.seq[target_pos]].setdefault(Otype, 0)
    target_pos_ratio[record.seq[target_pos]][Otype] += 1

target_res = sorted(target_pos_ratio.items(), key=lambda item: sum(item[1].values()), reverse=True)
#print(target_res[0])
    
###

outermost_grid = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.2)

###



right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], subplot_spec=outermost_grid[1])


rx1 = plt.subplot(right_grid[0])
first_res = target_res[0]
O_stats = sorted(first_res[1].items(), key=lambda item: item[1], reverse=True)
#print(O_stats)

def my_autopct_1(pct):
    return ('%1.1f%%' % pct) if pct >= 1.5 else ''

rx1.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 6 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels
        shadow=True,
        pctdistance=0.80,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_1,
        explode=[0.1]*len(O_stats),
        startangle=90,
       )

rx2 = plt.subplot(right_grid[1])
second_res = target_res[1]

O_stats = sorted(second_res[1].items(), key=lambda item: item[1], reverse=True)

def my_autopct_2(pct):
    return ('%1.1f%%' % pct) if pct >= 3.0 else ''

rx2.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 7 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels 
        shadow=True,
        pctdistance=0.8,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_2,
        explode=[0.2]*len(O_stats),
        startangle=90,
       )


###

outer_grid = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[6, 1, 1.5], subplot_spec=outermost_grid[0], hspace=0.2)

   

#plt.style.use('default')

inner_grid = gridspec.GridSpecFromSubplotSpec(nrows=len(OtypeList), ncols=2, subplot_spec=outer_grid[0], width_ratios=[1, 1])

#axs1, axs2, axs3 = plt.subplot(inner_grid[0]), plt.subplot(inner_grid[1]), plt.subplot(inner_grid[2])
#fig, axs = ax1.subplots(nrows=len(OtypeList), ncols=2, figsize=[7,4], gridspec_kw={'width_ratios': [1, 1]})
highlights = [[1,4,8],
              [3,4,8,9],
             ]

for idx, Otype in enumerate(OtypeList):
    for i in range(2):
        #ax = fig.add_subplot()
        #ax = axs[idx,i]
        #ax = fig.add_subplot(gs[idx, i])
        ax = plt.subplot(inner_grid[idx, i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if i == 0:
            ax.set_ylabel(OtypeList[idx], rotation="horizontal")
            ax.yaxis.set_label_coords(-0.1, 0.2)
        target_alignment = [seqs[i] for seqs in  msa_group_by[Otype]]
        #print(msa_group_by[Otype])
        counts_mat = lm.alignment_to_matrix(target_alignment)
        aa_logo = lm.Logo(counts_mat, color_scheme="charge", ax=ax)
        #aa_logo.style_xticks(anchor=0, spacing=5)
        for position in highlights[i]:
            aa_logo.highlight_position(p=position, color="gold", alpha=.7)

###
### InterProScan 결과 그리는 부분 ###
###

features = []

with open("MSA/{}.interproscan.tsv".format(target_file), "r") as infile:
    for line in infile:
        fields = line.strip().split("\t")
        feat_name = fields[3]+" - "+fields[5]
        #print(feat_name)
        if "bind" not in feat_name:
            if fields[3] not in ["ProSitePatterns"]: continue #Pfam
        _start, _end = int(fields[6]), int(fields[7])
        
        features.append(GraphicFeature(start=_start, end=_end, strand=0,
                                       label = feat_name
        ))

ax2 = plt.subplot(outer_grid[2])
record = GraphicRecord(sequence_length=max(df_max_composition['Position']), features=features)
record.plot(ax=ax2)#, figure_width=10)

ax3 = plt.subplot(outer_grid[1])
ax3.plot(df_max_composition['Position'], df_max_composition['Max Composition'], marker='', color="black", alpha=0.9, linewidth=1)
#ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.margins(x=0)
ax3.set_ylim(0.5,1)


# #### WecC

# In[890]:


target_file = "201_1" #684_0, 117_3, 66_0

am = AlignIO.read("MSA/{}.addfrag.mafft.msa".format(target_file), "fasta")

###

first_seq = am[0]
positions_to_remove = [pos for pos, char in enumerate(first_seq) if char == '-']
new_alignment = MultipleSeqAlignment([])
for record in am:
    new_seq = "".join([char for pos, char in enumerate(record.seq) if pos not in positions_to_remove])
    new_record = SeqRecord(Seq(new_seq), id=record.id, description=record.description)
    new_alignment.append(new_record)
    
am = new_alignment

###

alignment_length = am.get_alignment_length()
aa_composition = []

for i in range(alignment_length):
    column = am[:, i]
    aa_counts = {}
    total_count = len(column)

    for aa in column:
        if aa not in aa_counts:
            aa_counts[aa] = 1
        else:
            aa_counts[aa] += 1

    # Calculate the frequency of each amino acid
    aa_frequencies = {aa: count / total_count for aa, count in aa_counts.items()}
    aa_composition.append(aa_frequencies)
    
###

max_composition = [max(position.values()) for position in aa_composition]
df_max_composition = pd.DataFrame({
    'Position': range(1, len(max_composition) + 1),
    'Max Composition': max_composition
})


# In[891]:


target_file = "201_1"

plt.figure(figsize=(16, 9))
target_pos = 48
OtypeList = ["O157", "O26", "O103", "O8", "O121", "O111", "O25", "O149", "O145", "O9"]

### 
### 제일 윗부분 LOGO 그리는 부분 ###
### + 위치별 통계
###

msa_group_by = {}
target_pos_ratio = {}
for record in am:
    Otype = record.id.split(":")[1]
    msa_group_by.setdefault(Otype, [])
    msa_group_by[Otype].append((str(record.seq[40:50]), str(record.seq[270:280])))
    
    target_pos_ratio.setdefault(record.seq[target_pos], {})
    target_pos_ratio[record.seq[target_pos]].setdefault(Otype, 0)
    target_pos_ratio[record.seq[target_pos]][Otype] += 1

target_res = sorted(target_pos_ratio.items(), key=lambda item: sum(item[1].values()), reverse=True)
#print(target_res[0])
    
###

outermost_grid = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.2)

###



right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[1, 1], subplot_spec=outermost_grid[1])


rx1 = plt.subplot(right_grid[0])
first_res = target_res[0]
O_stats = sorted(first_res[1].items(), key=lambda item: item[1], reverse=True)
#print(O_stats)

def my_autopct_1(pct):
    return ('%1.1f%%' % pct) if pct >= 1.5 else ''

rx1.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 6 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels
        shadow=True,
        pctdistance=0.80,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_1,
        explode=[0.1]*len(O_stats),
        startangle=90,
       )

rx2 = plt.subplot(right_grid[1])
second_res = target_res[1]

O_stats = sorted(second_res[1].items(), key=lambda item: item[1], reverse=True)

def my_autopct_2(pct):
    return ('%1.1f%%' % pct) if pct >= 3.0 else ''

rx2.pie([stats[1] for stats in O_stats], # Sizes
        labels=[lb+" (n={})".format(O_stats[idx][1]) if idx < 7 else "" for idx, lb in enumerate(stats[0] for stats in O_stats)], # Labels 
        shadow=True,
        pctdistance=0.8,
        colors=plt.cm.Pastel1.colors,
        autopct=my_autopct_2,
        explode=[0.2]*len(O_stats),
        startangle=90,
       )


###

outer_grid = gridspec.GridSpecFromSubplotSpec(3, 1, height_ratios=[6, 1, 1.5], subplot_spec=outermost_grid[0], hspace=0.2)

   

#plt.style.use('default')

inner_grid = gridspec.GridSpecFromSubplotSpec(nrows=len(OtypeList), ncols=2, subplot_spec=outer_grid[0], width_ratios=[1, 1])

#axs1, axs2, axs3 = plt.subplot(inner_grid[0]), plt.subplot(inner_grid[1]), plt.subplot(inner_grid[2])
#fig, axs = ax1.subplots(nrows=len(OtypeList), ncols=2, figsize=[7,4], gridspec_kw={'width_ratios': [1, 1]})
highlights = [[1,4,8],
              [3,4,8,9],
             ]

for idx, Otype in enumerate(OtypeList):
    for i in range(2):
        #ax = fig.add_subplot()
        #ax = axs[idx,i]
        #ax = fig.add_subplot(gs[idx, i])
        ax = plt.subplot(inner_grid[idx, i])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if i == 0:
            ax.set_ylabel(OtypeList[idx], rotation="horizontal")
            ax.yaxis.set_label_coords(-0.1, 0.2)
        target_alignment = [seqs[i] for seqs in  msa_group_by[Otype]]
        #print(msa_group_by[Otype])
        counts_mat = lm.alignment_to_matrix(target_alignment)
        aa_logo = lm.Logo(counts_mat, color_scheme="charge", ax=ax)
        #aa_logo.style_xticks(anchor=0, spacing=5)
        for position in highlights[i]:
            aa_logo.highlight_position(p=position, color="gold", alpha=.7)

###
### InterProScan 결과 그리는 부분 ###
###

features = []

with open("MSA/{}.interproscan.tsv".format(target_file), "r") as infile:
    for line in infile:
        fields = line.strip().split("\t")
        feat_name = fields[3]+" - "+fields[5]
        #print(feat_name)
        #if fields[3] not in ["ProSitePatterns", "Pfam"]: continue #Pfam
        if "bind" not in feat_name:
            if fields[3] not in ["ProSitePatterns"]:
                continue
        _start, _end = int(fields[6]), int(fields[7])
        
        features.append(GraphicFeature(start=_start, end=_end, strand=0,
                                       label = feat_name
        ))

ax2 = plt.subplot(outer_grid[2])
record = GraphicRecord(sequence_length=max(df_max_composition['Position']), features=features)
record.plot(ax=ax2)#, figure_width=10)

ax3 = plt.subplot(outer_grid[1])
ax3.plot(df_max_composition['Position'], df_max_composition['Max Composition'], marker='', color="black", alpha=0.9, linewidth=1)
#ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.fill_between(df_max_composition['Position'], df_max_composition['Max Composition'], color='skyblue', alpha=0.8)
ax3.margins(x=0)
ax3.set_ylim(0.5,1)


# In[ ]:





# In[ ]:





# ### 11. Make model with all O serotype-related genes 

# In[618]:


DataMat_list = []
dimVec = 320
temp_vecs = {}

for clstrMBGD, clstrCDHIT in target_clstr:
    DataMat = {"acc":[], "answer":[]}
    for i in range(dimVec):
        DataMat["MBGD{}_CDHIT{}_v{}".format(clstrMBGD, clstrCDHIT, i)] = []        
        
    with open("MBGD_clusters_cdhit/{}/{}_{}.vecs".format(int(clstrMBGD)%1000, clstrMBGD, clstrCDHIT), "r") as infile:
        for line in infile:
            if line.startswith(">"): # odd lines
                acc, O, H = line.strip("\n").strip(">").split(":")[:3]
                continue
            else:
                sample_acc = acc.split("|")[0].split(".")[0].replace("_genomic", "")
                if sample_acc in OHtable:
                    vecs = line.strip("\n").split(" ")
                    temp_vecs.setdefault(sample_acc, [])
                    temp_vecs[sample_acc].append([float(vec) for vec in vecs])
                    
        duplicate_count = 0
        for sample_acc in temp_vecs:
            mean_vec = np.mean(temp_vecs[sample_acc], axis = 0)
            DataMat["acc"].append(sample_acc)
            DataMat["answer"].append(OHtable[sample_acc][0])
            for i in range(len(mean_vec)):
                DataMat["MBGD{}_CDHIT{}_v{}".format(clstrMBGD, clstrCDHIT, i)].append(mean_vec[i])
            if len(temp_vecs[sample_acc]) > 1: duplicate_count += 1
        #print(duplicate_count)
        
    DataMat_list.append(pd.DataFrame(DataMat))


# In[1543]:


print(duplicate_count)


# In[1556]:


DataMat_list[0]


# In[1555]:


DataMat_allgene = DataMat_list[1]

for i in [j for j in range(2, len(DataMat_list))]:
    DataMat_allgene = DataMat_allgene.merge(DataMat_list[i].drop('answer', axis=1), left_on = "acc", right_on = "acc")

DataMat_allgene


# In[1564]:


DataMat_allgene_1911 = pd.merge(DataMat_allgene, DataMat_list[0].drop('answer', axis=1), on='acc', how='left').fillna(0)


# In[1565]:


DataMat_allgene_1911


# In[1603]:


dict(DataMat_allgene["answer"].value_counts())["O121"]


# In[1566]:


#Y = DataMat_allgene["answer"].astype("category")
#X = DataMat_allgene.drop(["answer", "acc"], axis=1)
Y = DataMat_allgene_1911["answer"].astype("category")
X = DataMat_allgene_1911.drop(["answer", "acc"], axis=1)

RF_estimator = RandomForestClassifier(n_estimators=100, random_state=int(time.time()), n_jobs=40, criterion="entropy", max_depth=10, class_weight="balanced")
scores = cross_val_score(RF_estimator, X, Y, cv=StratifiedKFold(n_splits = 10))

print(scores.mean(), scores.std())


# In[ ]:





# ### 12. Feature Importance for elucidating key gene for O serotype determination

# In[1253]:


nFold = 5.0
target_O_list = [label for label, count in dict(DataMat_allgene["answer"].value_counts()).items() if count >= nFold]

DataMat_test = DataMat_allgene
DataMat_test = DataMat_test[DataMat_test["answer"].isin(target_O_list)]


# In[1391]:


Y = DataMat_test["answer"].astype("category")
X = DataMat_test.drop(["answer", "acc"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
RF_estimator.fit(x_train, y_train)


# In[1392]:


print(metrics.classification_report(list(y_test), list(RF_estimator.predict(x_test))))


# In[1256]:


start_time = time.time()
importances = RF_estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF_estimator.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[1257]:


feature_names = DataMat.drop(["answer", "acc"], axis=1).columns
forest_importances = pd.Series(importances, index=feature_names)
top15 = forest_importances.sort_values(ascending=False)[:15]

top15


# In[1258]:


[ClusterMeta[a.split("_")[0].strip("MBGD")]["gene"]+"_"+a.split("_")[-1] for a in list(top15.index)]


# In[1668]:


plt.figure(figsize=(1.5,5))
plt.title('Feature Importances Top 15', fontsize=7)
#sns.barplot(x=top15, y=top15.index)
sns.barplot(x=top15, y=[ClusterMeta[a.split("_")[0].strip("MBGD")]["gene"]+"_"+a.split("_")[-1] for a in list(top15.index)])
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.show()


# In[1259]:


major_O_list = list(dict(meta["O"].value_counts()).keys())[:5]
sf_cfm = metrics.confusion_matrix(list(y_test), list(RF_estimator.predict(x_test)), labels=major_O_list)


# In[1260]:


O_count_sums = np.array([dict(y_test.value_counts())[Otype] for Otype in major_O_list])
sf_cfm_rt = sf_cfm / O_count_sums[:, np.newaxis]


# In[1636]:


Y_predict = cross_val_predict(RF_estimator, X, Y, cv=StratifiedKFold(n_splits = 10))


# In[1597]:


minor_O_list = list(dict(meta["O"].value_counts()).keys())[120:125]
minor_O_list


# In[1637]:


Y


# In[1638]:


Y_predict


# In[1639]:


cf_major = confusion_matrix(Y, Y_predict, labels=major_O_list)
cf_minor = confusion_matrix(Y, Y_predict, labels=minor_O_list)


# In[1605]:


major_O_count_sums = np.array([dict(Y.value_counts())[Otype] for Otype in major_O_list])
minor_O_count_sums = np.array([dict(Y.value_counts())[Otype] for Otype in minor_O_list])
major_O_count_sums
#minor_O_count_sums


# In[1606]:


major_O_count_sums[4] += 1


# In[1513]:


cf
cf / major_O_count_sums[:, np.newaxis]


# In[1640]:


cf_major


# In[1608]:


# 혼동 행렬을 시각화하는 함수
def plot_confusion_matrix(ax, cm, labels, sum_value, title, ylabel, xlabel):
    #plt.figure(figsize=(6, 4))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=8)
    #plt.colorbar(cax, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=8)

    # 텍스트 표시
    fmt = ""#'.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j] if i!=j else str(cm[i, j])+"\n("+str(sum_value[i])+")", fmt),
                 ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    #plt.tight_layout()

# 레이블 예시
labels = ['Class 1', 'Class 2', 'Class 3']

# 혼동 행렬 플롯
fig, ax = plt.subplots(2, 1, figsize=(2, 5))
#plot_confusion_matrix(ax[0], major_cm_rt, major_O_list, "Recall for SerotypeFinder\nMajor O types")
#plot_confusion_matrix(ax[0], cf_major / major_O_count_sums[:, np.newaxis], major_O_list, "Recall for RF model\n(Major O types)", "True label", "")
plot_confusion_matrix(ax[0], cf_major, major_O_list, major_O_count_sums, "Confusion Matrix for RF model\n(Major O types)", "True label", "")
#plot_confusion_matrix(ax[1], middle_cm_rt, middle_O_list, "Recall for SerotypeFinder\nMinor O types")
#plot_confusion_matrix(ax[1], cf_minor / minor_O_count_sums[:, np.newaxis], minor_O_list, "(Minor O types)", "Tru e label", "Predicted label")
plot_confusion_matrix(ax[1], cf_minor, minor_O_list, minor_O_count_sums, "(Minor O types)", "True label", "Predicted label")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[1412]:


major_O_count_sums


# In[1413]:


X


# In[ ]:





# In[ ]:





# In[1261]:


# 혼동 행렬을 시각화하는 함수
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=8)
    plt.yticks(tick_marks, labels, fontsize=8)

    # 텍스트 표시
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)
    plt.tight_layout()

# 레이블 예시
labels = ['Class 1', 'Class 2', 'Class 3']

# 혼동 행렬 플롯
plot_confusion_matrix(sf_cfm_rt, major_O_list)
plt.show()


# In[1393]:


major_O_list = list(dict(meta["O"].value_counts()).keys())[0:5]
#major_O_list.remove("O111")
middle_O_list = list(dict(meta["O"].value_counts()).keys())[100:105]
major_cm = metrics.confusion_matrix(list(y_test), list(RF_estimator.predict(x_test)), labels=major_O_list)
middle_cm = metrics.confusion_matrix(list(y_test), list(RF_estimator.predict(x_test)), labels=middle_O_list)

major_O_count_sums = np.array([dict(y_test.value_counts())[Otype] for Otype in major_O_list])
middle_O_count_sums = np.array([dict(y_test.value_counts())[Otype] for Otype in middle_O_list])
major_cm_rt = major_cm / major_O_count_sums[:, np.newaxis]
middle_cm_rt = middle_cm / middle_O_count_sums[:, np.newaxis]

# 혼동 행렬을 시각화하는 함수
def plot_confusion_matrix(ax, cm, labels, title):
    #plt.figure(figsize=(6, 4))
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=9)
    #plt.colorbar(cax, ax=ax)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=8)

    # 텍스트 표시
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center", fontsize=6,
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=8)
    ax.set_xlabel('Predicted label', fontsize=8)
    #plt.tight_layout()

# 레이블 예시
labels = ['Class 1', 'Class 2', 'Class 3']

# 혼동 행렬 플롯
fig, ax = plt.subplots(2, 1, figsize=(2, 5))
#plot_confusion_matrix(ax[0], major_cm_rt, major_O_list, "Recall for RF model\nMajor O types")
plot_confusion_matrix(ax[0], major_cm_rt, major_O_list, "Recall for RF model")
#plot_confusion_matrix(ax[1], middle_cm_rt, middle_O_list, "Recall for RF model\nMinor O types")
plot_confusion_matrix(ax[1], middle_cm_rt, middle_O_list, "")
plt.show()


# ### 13. Test for 9-gene GNB, SVC, MLP, RF, AdaBoost, XGB model

# In[1579]:


#nFold = 10
#target_O_list = [label for label, count in dict(DataMat["answer"].value_counts()).items() if count >= nFold]
target_O_list = [label for label, count in dict(DataMat_allgene_1911["answer"].value_counts()).items() if count >= 2]
#DataMat_test = DataMat[DataMat["answer"].isin(target_O_list)]
DataMat_test = DataMat_allgene_1911[DataMat_allgene_1911["answer"].isin(target_O_list)]


# In[1581]:


len(target_O_list)


# In[1582]:


Y = DataMat_test["answer"]
X = DataMat_test.drop(["answer", "acc"], axis=1)
#Y = DataMat_allgene_1911["answer"]
#X = DataMat_allgene_1911.drop(["answer", "acc"], axis=1)

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
RF_estimator = RandomForestClassifier(n_estimators=100, random_state=33, n_jobs=40, criterion="entropy", max_depth=10, class_weight="balanced")
GNB_estimator = GaussianNB()
SVC_estimator = svm.SVC(kernel="linear", class_weight="balanced")
MLP_estimator = MLPClassifier(solver='adam', random_state=0, hidden_layer_sizes=[500])
KNN_estimator = KNeighborsClassifier(n_jobs=40)
#ADB_estimator = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=50, random_state=33, class_weight="balanced"),
#                                   n_estimators=100, random_state=33,)
XGB_estimator = xgb.XGBClassifier(n_jobs=40, max_depth=10, eta=0.1)

models_metric = {}
for model, estimator in {"RF": RF_estimator, "XGB": XGB_estimator, "GNB": GNB_estimator, "MLP": MLP_estimator, "SVC": SVC_estimator, "KNN": KNN_estimator
                         }.items():
                        #"ADB": ADB_estimator,
    #scoring_auroc = make_scorer(roc_auc_score, multi_class='ovo')#,needs_proba=True)
    #scores_auroc = cross_validate(estimator, X, Y, cv=StratifiedKFold(n_splits = nFold), scoring=scoring_auroc)

    metrics = {
        "f1" : [],
        "recall" : [],
        "prc": [],
        "acc": [],
        #"auroc": scores_auroc
    }
    
    le = LabelEncoder()    
    Y_trans = le.fit_transform(Y) if model == "XGB" else Y
    
    #for train_idx, test_idx in StratifiedKFold(n_splits = nFold).split(X, Y_trans):
    for train_idx, test_idx in StratifiedShuffleSplit(n_splits = nFold).split(X, Y_trans):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        if model == "XGB": y_train, y_test = Y_trans[train_idx], Y_trans[test_idx]
        else: y_train, y_test = Y_trans.iloc[train_idx], Y_trans.iloc[test_idx]        

        clf = estimator.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics["f1"].append(f1_score(y_test, y_pred, average="weighted"))
        metrics["recall"].append(recall_score(y_test, y_pred, average="weighted"))
        metrics["prc"].append(precision_score(y_test, y_pred, average="weighted"))
        metrics["acc"].append(accuracy_score(y_test, y_pred))
    
    models_metric[model] = metrics


# In[1583]:


models_metric


# In[1708]:


#np.mean(models_metric["SVC"]["acc"])
#np.std(models_metric["SVC"]["acc"])


# In[1575]:


models_metric_save = models_metric


# In[1576]:


with open('models_metric_previous1911.json', 'w') as json_file:
    json.dump(models_metric_save, json_file)


# In[1709]:


sf_report


# In[941]:


ec_report


# In[1585]:


models = list(models_metric.keys())

plt.style.use('default')
fig = plt.figure(figsize=(7,4))

N = len(models_metric) + 2
ind = np.arange(N)  
#width = 0.25
width = 0.15
hex_code = sns.color_palette("pastel6").as_hex()
  
#acc_vals_mean = [np.mean(models_metric[model]["acc"]) for model in models] 
acc_vals_mean = [np.mean(models_metric[model]["acc"]) for model in models] + [sf_report.loc["accuracy"]["precision"], ec_report.loc["accuracy"]["precision"]]
acc_vals_std = [np.std(models_metric[model]["acc"]) for model in models] + [0, 0]
bar1 = plt.bar(ind-1.5*width, acc_vals_mean, width, color = hex_code[2], edgecolor="black") 
bar1_err = plt.errorbar(ind-1.5*width, acc_vals_mean, acc_vals_std, color = "grey", alpha=0.5, fmt="o", markersize=1.5, elinewidth=1) 
#bar1 = plt.errorbar(ind-width, acc_vals, width = width, color = hex_code[5]) 

prc_vals_mean = [np.mean(models_metric[model]["prc"]) for model in models] + [sf_report.loc["weighted avg"]["precision"], ec_report.loc["weighted avg"]["precision"]]
prc_vals_std = [np.std(models_metric[model]["prc"]) for model in models] + [0, 0]
bar2 = plt.bar(ind-0.5*width, prc_vals_mean, width, color=hex_code[0], edgecolor="black") 
bar2_err = plt.errorbar(ind-0.5*width, prc_vals_mean, prc_vals_std, color = "grey", alpha=0.5, fmt="o", markersize=1.5, elinewidth=1) 

rec_vals_mean = [np.mean(models_metric[model]["recall"]) for model in models] + [sf_report.loc["weighted avg"]["recall"], ec_report.loc["weighted avg"]["recall"]]
rec_vals_std = [np.std(models_metric[model]["recall"]) for model in models] + [0, 0]
bar3 = plt.bar(ind+0.5*width, rec_vals_mean, width, color=hex_code[5], edgecolor="black") 
bar3_err = plt.errorbar(ind+0.5*width, rec_vals_mean, rec_vals_std, color = "grey", alpha=0.5, fmt="o", markersize=1.5, elinewidth=1) 

f1_vals_mean = [np.mean(models_metric[model]["f1"]) for model in models] + [sf_report.loc["weighted avg"]["f1-score"], ec_report.loc["weighted avg"]["f1-score"]]
f1_vals_std = [np.std(models_metric[model]["f1"]) for model in models] + [0, 0]
bar4 = plt.bar(ind+1.5*width, f1_vals_mean, width, color = hex_code[3], edgecolor="black") 
bar4_err = plt.errorbar(ind+1.5*width, f1_vals_mean, f1_vals_std, color = "grey", alpha=0.5, fmt="o", markersize=1.5, elinewidth=1) 

#plt.xlabel("Models") 
plt.ylim((0.5, 1)) 
#plt.ylabel('') 
plt.title("") 

plt.xticks(ind, models + ["Serotype\nFinder", "ECTyper"], fontsize=9, rotation=0) 
plt.legend( (bar1, bar2, bar3, bar4), ('accuracy', 'precision', "recall", 'f1'), loc='upper left', bbox_to_anchor=(1,1))
plt.grid(axis="y", alpha=0.5)
plt.style.use('default')
#plt.tight_layout()
plt.show() 


# In[ ]:


Y = DataMat_test["answer"]
X = DataMat_test.drop(["answer", "acc"], axis=1)

rfecv = RFECV(estimator=RF_estimator, step=1, cv=StratifiedKFold(n_splits = nFold), scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X, Y)

#여기


# In[1440]:


rfecv


# In[1460]:


combined = dict(zip(list(X.columns[rfecv.get_support()]), list(rfecv.estimator_.feature_importances_)))
sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)


# In[1461]:


sorted_combined


# In[1462]:


sorted_combined[:15]


# In[1466]:


plt.figure(figsize=(3,5))
plt.title('Feature Importances Top 15')
#sns.barplot(x=top15, y=top15.index)
#sns.barplot(x=top15, y=[ClusterMeta[a.split("_")[0].strip("MBGD")]["gene"]+"_"+a.split("_")[-1] for a in list(top15.index)])
sns.barplot(x=[comp[1] for comp in sorted_combined[:15]],
            y=[ClusterMeta[comp[0].split("_")[0].strip("MBGD")]["gene"]+"_"+comp[0].split("_")[-1]
               for comp in sorted_combined[:15]])
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()


# In[ ]:





# In[56]:


start_time = time.time()
importances = RF_estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF_estimator.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")


# In[ ]:





# In[ ]:





# ### 10. Re-map the genes to 19-gene DB

# In[26]:


fl = glob.glob("2-1.split/overall_reduced_DB2/*.tab")
target_acc = {}

for af in fl:
    with open(af, "r") as infile:
        file_name = af.split("/")[-1].replace(".tab", "")
        
        for line in infile:
            seqnum, gene, pident, length = line.strip("\n").split("\t")[0:4]
            _sp, _gene = gene.split(":")
            if _sp in Gene2Cluster:
                if _gene in Gene2Cluster[_sp]:
                    clstr_cand = Gene2Cluster[_sp][_gene]
                    for clstr in clstr_cand:
                        if int(clstr) in target_clstr:
                            target_acc.setdefault("{}|{}".format(file_name, seqnum), [])
                            target_acc["{}|{}".format(file_name, seqnum)].append(clstr)


# In[27]:


table = {}

for sample in [af.split("/")[-1].replace(".tab", "") for af in glob.glob("2-1.split/overall_reduced_DB/*.tab")]:
    for clstr in target_clstr:
        table.setdefault(sample, {})
        table[sample].setdefault(clstr, [])

for k, v in target_acc.items():
    sample = k.split("|")[0]
    if sample in table:
        for clstr in v:
            if int(clstr) in table[sample]:
                table[sample][int(clstr)].append(k)


# In[28]:


table_v2 = {}

with open("5-1.ESM/GMS2.faa.txt", "r") as infile:
    for line in infile:
        if line.startswith(">"):
            acc = line.strip("\n").strip(">")
            if acc in target_acc:
                read = True
            else:
                read = False
                
        elif read:
            vecs = [float(vec) for vec in line.strip("\n").split(" ")]
            table_v2[acc] = vecs           


# Make Dataframe for Learning

# In[29]:


DataMat_train = {"{}_v{}".format(j, i):[] for i in range(320) for j in target_clstr}
DataMat_train["acc"] = []
DataMat_train["entero_acc"] = []
DataMat_train["answer"] = []

for _sp in table:
    try:
        DataMat_train["answer"].append(meta[meta["acc"] == _sp.split(".")[0].split("_genomic")[0]]["O"].item())
    except ValueError: # 
        continue
    DataMat_train["acc"].append(_sp)
    DataMat_train["entero_acc"].append(_sp.split(".")[0].split("_genomic")[0])
    
    for clstr in target_clstr:
    #for clstr in [117, 201]:
        vecs = [-1.0]*320
        if clstr in table[_sp]:
            accs = table[_sp][clstr]
            accs_vec = [np.array(table_v2[acc]) for acc in accs if acc in table_v2]
            if len(accs_vec) == 0:
                vecs = np.array([-1.0]*320)
            else:
                vecs = np.mean(accs_vec, axis = 0)            
        
        for i in range(320):
            DataMat_train["{}_v{}".format(clstr, i)].append(vecs[i])

DataMat_train = pd.DataFrame(DataMat_train)


# In[30]:


#DataMat = DataMat_train[["{}_v{}".format(j, i) for i in range(320) for j in [7478]] + ["acc", "entero_acc", "answer"]]
#DataMat = DataMat_train[DataMat_train["answer"].isin(major_O_list)]
DataMat = DataMat_train


# In[31]:


DataMat


# In[32]:


DataMat["3039_v319"].value_counts()#[-1.0]


# ### 11. Test for GNB, SVC, MLP, RF, AdaBoost, XGB model

# In[ ]:


nFold = 3
target_O_list = [label for label, count in dict(DataMat["answer"].value_counts()).items() if count >= nFold]
DataMat_test = DataMat[DataMat["answer"].isin(target_O_list)]

Y = DataMat_test["answer"]
X = DataMat_test.drop(["answer", "acc", "entero_acc"], axis=1)

#estimator = svm.SVC(kernel="linear", class_weight="balanced")
#estimator = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=50, random_state=33, class_weight="balanced"),
#                                   n_estimators=100, random_state=33,)
#estimator = KNeighborsClassifier(n_jobs=40)
#estimator = LogisticRegression(multi_class="ovr", solver="lbfgs", class_weight="balanced", n_jobs=40)
for train_idx, test_idx in StratifiedKFold(n_splits = nFold).split(X, Y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
    
    clf = estimator.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))


# In[ ]:


nFold = 10
target_O_list = [label for label, count in dict(DataMat["answer"].value_counts()).items() if count >= nFold]
DataMat_test = DataMat[DataMat["answer"].isin(target_O_list)]

Y = DataMat_test["answer"]
X = DataMat_test.drop(["answer", "acc", "entero_acc"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
RF_estimator = RandomForestClassifier(n_estimators=100, random_state=33, n_jobs=40, criterion="entropy", max_depth=10, class_weight="balanced")
GNB_estimator = GaussianNB()
SVC_estimator = svm.SVC(kernel="linear", class_weight="balanced")
MLP_estimator = MLPClassifier(solver='adam', random_state=0, hidden_layer_sizes=[500])
KNN_estimator = KNeighborsClassifier(n_jobs=40)
#ADB_estimator = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=50, random_state=33, class_weight="balanced"),
#                                   n_estimators=100, random_state=33,)
XGB_estimator = xgb.XGBClassifier(n_jobs=40, max_depth=10, eta=0.1)

models_metric = {}
for model, estimator in {"RF": RF_estimator, "XGB": XGB_estimator, #"GNB": GNB_estimator, "MLP": MLP_estimator, "SVC": SVC_estimator, "KNN": KNN_estimator
                         }.items():
                        #"ADB": ADB_estimator,
    #scoring_auroc = make_scorer(roc_auc_score, multi_class='ovo')#,needs_proba=True)
    #scores_auroc = cross_validate(estimator, X, Y, cv=StratifiedKFold(n_splits = nFold), scoring=scoring_auroc)

    metrics = {
        "f1" : [],
        "prc": [],
        "acc": [],
        #"auroc": scores_auroc
    }
    
    le = LabelEncoder()    
    Y_trans = le.fit_transform(Y) if model == "XGB" else Y
    
    for train_idx, test_idx in StratifiedKFold(n_splits = nFold).split(X, Y_trans):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        
        if model == "XGB": y_train, y_test = Y_trans[train_idx], Y_trans[test_idx]
        else: y_train, y_test = Y_trans.iloc[train_idx], Y_trans.iloc[test_idx]        

        clf = estimator.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics["f1"].append(f1_score(y_test, y_pred, average="weighted"))
        metrics["prc"].append(precision_score(y_test, y_pred, average="weighted"))
        metrics["acc"].append(accuracy_score(y_test, y_pred))
    
    models_metric[model] = metrics


# In[39]:


models_metric


# In[44]:


models_metric_save = models_metric


# In[38]:


models = list(models_metric.keys())

plt.style.use('default')
fig = plt.figure(figsize=(6,6))

N = len(models_metric)
ind = np.arange(N)  
width = 0.25
hex_code = sns.color_palette("pastel6").as_hex()
  
acc_vals_mean = [np.mean(models_metric[model]["acc"]) for model in models] 
acc_vals_std = [np.std(models_metric[model]["acc"]) for model in models] 
bar1 = plt.bar(ind-width, acc_vals_mean, width, color = hex_code[5]) 
bar1_err = plt.errorbar(ind-width, acc_vals_mean, acc_vals_std, color = "grey", fmt="o", markersize=1.5, elinewidth=1) 
#bar1 = plt.errorbar(ind-width, acc_vals, width = width, color = hex_code[5]) 

prc_vals_mean = [np.mean(models_metric[model]["prc"]) for model in models] 
prc_vals_std = [np.std(models_metric[model]["prc"]) for model in models] 
bar2 = plt.bar(ind, prc_vals_mean, width, color=hex_code[0]) 
bar2_err = plt.errorbar(ind, prc_vals_mean, prc_vals_std, color = "grey", fmt="o", markersize=2.5, elinewidth=2) 
  
f1_vals = [np.mean(models_metric[model]["f1"]) for model in models] 
bar3 = plt.bar(ind+width, f1_vals, width, color = hex_code[3]) 

plt.xlabel("Models") 
plt.ylim((0.6, 1)) 
#plt.ylabel('') 
plt.title("Players Score") 
  
plt.xticks(ind, models) 
plt.legend( (bar1, bar2, bar3), ('accuracy', 'precision', 'f1'))
plt.style.use('default')
plt.show() 


# ### 12. Test for 19-gene RF model

# In[43]:


Y = DataMat_train["answer"]
X = DataMat_train.drop(["answer", "acc", "entero_acc"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
RF_estimator = RandomForestClassifier(n_estimators=100, random_state=33, n_jobs=40, criterion="entropy", max_depth=10, class_weight="balanced")

try:
    RF_estimator.fit(x_train, y_train)
except ValueError:
    exit()

y_pred = RF_estimator.predict(x_test)
print("Accuracy : {}".format(accuracy_score(y_test, y_pred)))


# In[33]:


Y.value_counts()[:20].keys()


# In[101]:


#major_labels = list(Y.value_counts()[150:].keys())
major_labels = list(Y.value_counts()[:10].keys())
print(metrics.confusion_matrix(y_test, y_pred, labels=major_labels))


# In[107]:


#plt.figure(figsize=(20,20))
#plt.plot(ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_test, y_pred, labels=major_labels),
#                               display_labels=list(major_labels)))
#ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_test, y_pred)).plot()
#ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(y_test, y_pred, labels=major_labels),
#                       display_labels=list(major_labels)).plot().figure_.savefig("confusion_major.png", dpi=200)
fig = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=major_labels, display_labels=list(major_labels)).figure_.savefig("confusion_major.png", dpi=200)
fig.show()


# In[109]:


1


# ### 13. Feature Importance for elucidating key gene for O serotype determination

# ### 11. Test for 19-gene GNB, SVC, MLP, RF, AdaBoost, XGB model

# In[37]:


feature_names = DataMat_train.drop(["answer", "acc", "entero_acc"], axis=1).columns
forest_importances = pd.Series(importances, index=feature_names)
top15 = forest_importances.sort_values(ascending=False)[:40]

top15


# In[45]:


plt.figure(figsize=(10,10))
plt.title('Feature Importances Top 40')
sns.barplot(x=top15, y=top15.index)
plt.show()


# In[ ]:





# #
# Check whether genes assigned to same cluster are similar or not
# 
# #

# In[106]:


table["ESC_RB8374AA_AS.result"][117]


# In[116]:


Gene2Cluster["ece"]["Z3194"]
Gene2Cluster["rle"]["RL_RS18950"]


# ##### BLAST result of 'ESC_RB8374AA_AS.result|241' -> cpsG in E.coli
# ##### BLAST result of 'ESC_RB8374AA_AS.result|242' -> mannose-1-phosphate guanylyltransferase/mannose-6-phosphate isomerase
# ##### BLAST result of 'ESC_RB8374AA_AS.result|2058' -> glucose-1-phosphate thymidylyltransferase RfbA in E.coli
# ##### BLAST result of 'ESC_RB8374AA_AS.result|3610' -> glucose-1-phosphate adenylyltransferase
# ##### BLAST result of 'ESC_RB8374AA_AS.result|3659' -> tryptophan--tRNA ligase in E.coli
# ##### BLAST result of 'ESC_RB8374AA_AS.result|4832' -> mannose-6-phosphate isomerase in E.coli

# In[169]:


table["ESC_RB8374AA_AS.result"][201]


# ##### BLAST result of 'ESC_RB8374AA_AS.result|228' -> SDR family oxidoreductase in E.coli
# ##### BLAST result of 'ESC_RB8374AA_AS.result|238' -> UDP-glucose 6-dehydrogenase in E.coli
# ##### BLAST result of 'ESC_RB8374AA_AS.result|2060' -> UDP-N-acetyl-D-mannosamine dehydrogenase in E.coli

# #
# Check Prevalance of each gene clusters in a genome
# 
# #

# In[176]:


abundance_genome = {}

for _acc in table:
    for clstr in target_clstr:
        abundance_genome.setdefault(clstr, [])
        abundance_genome[clstr].append(len(table[_acc][clstr]))


# In[183]:


for clstr in abundance_genome:
    print(clstr, ClusterMeta[str(clstr)]["gene"], 9561-abundance_genome[clstr].count(0), sep="\t")


# #
# blast serotypefinder's wzx sequences into MBGD database
# 
# #

# In[185]:


fl = glob.glob("serotypefinder_db_toMBGD.tab")
serotypefinder_sequences = {}

for af in fl:
    with open(af, "r") as infile:
        file_name = af.split("/")[-1].replace(".tab", "")
        
        for line in infile:
            seqnum, gene, pident, length = line.strip("\n").split("\t")[0:4]
            _sp, _gene = gene.split(":")
            if _sp in Gene2Cluster:
                if _gene in Gene2Cluster[_sp]:
                    clstr_cand = Gene2Cluster[_sp][_gene]
                    for clstr in clstr_cand:
                        if int(clstr) in target_clstr:
                            serotypefinder_sequences.setdefault(seqnum, [])
                            serotypefinder_sequences[seqnum].append(clstr)


# In[186]:


serotypefinder_sequences


# #
# Check each gene's classification performance based on their vector
# 
# #

# In[19]:


fl = [af for af in glob.glob("7.Self_test/MBGD_BLAST/*.tab") if "Eco_303" not in af and "FORC_031" not in af]
target_acc = {}
temp = []

for af in fl:
    with open(af, "r") as infile:
        file_name = af.split("/")[-1].replace(".tab", "")
        temp.append(file_name)
        
        for line in infile:
            seqnum, gene, pident, length = line.strip("\n").split("\t")[0:4]
            _sp, _gene = gene.split(":")
            if _sp in Gene2Cluster:
                if _gene in Gene2Cluster[_sp]:
                    clstr_cand = Gene2Cluster[_sp][_gene]
                    for clstr in clstr_cand:
                        if int(clstr) in target_clstr:
                            target_acc.setdefault("{}|{}".format(file_name, seqnum), [])
                            target_acc["{}|{}".format(file_name, seqnum)].append(clstr)


# In[20]:


table = {}

for sample in [af.split("/")[-1].replace(".tab", "") for af in fl]:
    for clstr in target_clstr:
        table.setdefault(sample, {})
        table[sample].setdefault(clstr, [])

for k, v in target_acc.items():
    sample = k.split("|")[0]
    if sample in table:
        for clstr in v:
            if int(clstr) in table[sample]:
                table[sample][int(clstr)].append(k)


# In[21]:


table_v2 = {}

with open("7.Self_test/GMS2_MFDStest_merged.faa.txt", "r") as infile:
    for line in infile:
        if line.startswith(">"):
            acc = line.strip("\n").strip(">")
            if acc in target_acc:
                read = True
            else:
                read = False
                
        elif read:
            vecs = [float(vec) for vec in line.strip("\n").split(" ")]
            #print(acc, vecs)
            #_sp, _seqnum = acc.split("|")
            table_v2[acc] = vecs           


# In[22]:


DataMat_valid = {"{}_v{}".format(j, i):[] for i in range(320) for j in target_clstr}
DataMat_valid["acc"] = []
DataMat_valid["entero_acc"] = []
DataMat_valid["answer"] = []
MFDS_answer_original = {    "FORC_028" : "O26",    "FORC_029" : "O104",    "FORC_032" : "O6",    "FORC_043" : "O104",    "FORC_044" : "O157",
    "FORC_069" : "O104",    "Eco_292" : "O103",    "Eco_302" : "O25",    "Eco_342" : "O26",    "Eco_386" : "O145",
    "Eco_389" : "O103",    "Eco_435" : "O6",    "Eco_459" : "O103",    "Eco_471" : "O26",    "Eco_541" : "O145",    "Eco_551" : "O6"}

MFDS_answer_additional = {    "Eco_019" : "O145",    "Eco_028" : "O145",    "Eco_039" : "O157",    "Eco_038" : "O157",    "Eco_036" : "O26",
    "Eco_021" : "O103",    "Eco_042" : "O103",    "Eco_040" : "O111",    "Eco_041" : "O111",    "Eco_022" : "O6",    "Eco_031" : "O6",
    "Eco_014" : "O25",    "Eco_013" : "O25",    "Eco_037" : "O55",    "Eco_011" : "O104",    "Eco_030" : "O104",    "Eco_004" : "O127",
    "Eco_026" : "O127",    "Eco_001" : "O115",    "Eco_023" : "O115",    "Eco_035" : "O115",    "Eco_007" : "O36",    "Eco_015" : "O36",
    "Eco_009" : "O76",    "Eco_020" : "O76",    "Eco_027" : "O76",    "Eco_003" : "O121",    "Eco_029" : "O121",    "Eco_012" : "O121",
    "Eco_005" : "O168",    "Eco_036" : "O168",    "Eco_043" : "O168",    "Eco_010" : "O88",    "Eco_017" : "O88",    "Eco_018" : "O88",
    "Eco_008" : "O51",    "Eco_033" : "O51",    "Eco_034" : "O51",    "Eco_006" : "O177",    "Eco_025" : "O177",    "Eco_032" : "O177",
    "Eco_002" : "O116",    "Eco_016" : "O116",}

MFDS_answer = copy.deepcopy(MFDS_answer_original)

MFDS_answer.update(MFDS_answer_additional)


# In[23]:


for _sp in table:
    DataMat_valid["acc"].append(_sp)
    DataMat_valid["entero_acc"].append("MFDS_test")
    DataMat_valid["answer"].append(MFDS_answer[_sp])
    
    for clstr in target_clstr:
    #for clstr in [117, 201]:
        vecs = [-1.0]*320
        if clstr in table[_sp]:
            accs = table[_sp][clstr]
            accs_vec = [np.array(table_v2[acc]) for acc in accs if acc in table_v2]
            if len(accs_vec) == 0:
                vecs = np.array([-1.0]*320)
            #if acc in table_v2:
            #    vecs = table_v2[acc]
            else:
                vecs = np.mean(accs_vec, axis = 0)            
        
        for i in range(320):
            DataMat_valid["{}_v{}".format(clstr, i)].append(vecs[i])

DataMat_valid = pd.DataFrame(DataMat_valid)


# In[24]:


DataMat = DataMat_train
#DataMat = DataMat_train.append(DataMat_valid)
#DataMat = DataMat[["{}_v{}".format(j, i) for i in range(320) for j in [117, 201]] + ["acc", "entero_acc", "answer"]]


# In[344]:


DataMat


# In[345]:


train_set = DataMat[DataMat["entero_acc"] != "MFDS_test"]
valid_set = DataMat[DataMat.acc.isin(MFDS_answer_original)]


# In[346]:


valid_set


# In[347]:


Y = train_set["answer"]
X = train_set.drop(["answer", "acc", "entero_acc"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
x_valid, y_valid = valid_set.drop(["answer", "acc", "entero_acc"], axis=1), valid_set["answer"]
RF_estimator = RandomForestClassifier(n_estimators=1000, random_state=33, n_jobs=40, criterion="entropy", max_depth=50, class_weight="balanced")

try:    
    RF_estimator.fit(x_train, y_train)
except ValueError:
    exit()

y_pred = RF_estimator.predict(x_test)
print("Accuracy : {}".format(metrics.accuracy_score(y_test, y_pred)))

y_pred = RF_estimator.predict(x_valid)
print("Accuracy : {}".format(metrics.accuracy_score(y_valid, y_pred)))
print(metrics.confusion_matrix(y_valid, y_pred))
#print(metrics.confusion_matrix(y_valid, y_pred, labels=valid_set["answer"].unique()))


# In[350]:


print(list(valid_set["answer"]))


# In[349]:


y_pred


# In[ ]:





# #
# Check each gene's classification performance based on their sequence similarity
# 
# #

# In[ ]:


for _acc in table:
    record_dict = SeqIO.to_dict(SeqIO.parse("2.GeneMarkS2+/{0}/{0}.faa".format(_acc), "fasta"))

    for clstr in target_clstr:
        with open("7.Self_test/geneDB/{}.faa".format(clstr), "a") as outfile:
            seqs_list = [name.split("|")[1] for name in table[_acc][clstr]]
            for seqnum in seqs_list:
                if len(list(meta[meta["acc"] == _acc.split(".")[0].strip("_genomic")]["O"])) != 1:
                    print(_acc)
                print(">{}|{}|{}".format(_acc, seqnum, list(meta[meta["acc"] == _acc.split(".")[0].strip("_genomic")]["O"])[0]), file=outfile)
                #print(_acc, clstr, seqnum)
                print(record_dict[seqnum].seq, file=outfile)        


# In[ ]:





# In[ ]:





# In[191]:


table["ESC_RB8374AA_AS.result"][641]


# In[199]:


table["ESC_LB5116AA_AS.result"]


# In[ ]:





# In[ ]:





# In[44]:


DataMat_list = []
dimVec = 320
temp_vecs = {}

for clstrMBGD, clstrCDHIT in [(1911, 0), (5807, 5), (5807, 12)]:
    DataMat = {"acc":[], "answer":[]}
    for i in range(dimVec):
        DataMat["MBGD{}_CDHIT{}_v{}".format(clstrMBGD, clstrCDHIT, i)] = []        
        
    with open("MBGD_clusters_cdhit/{}/{}_{}.vecs".format(int(clstrMBGD)%1000, clstrMBGD, clstrCDHIT), "r") as infile:
        for line in infile:
            if line.startswith(">"): # odd lines
                acc, O, H = line.strip("\n").strip(">").split(":")[:3]
                continue
            else:
                sample_acc = acc.split("|")[0].split(".")[0].replace("_genomic", "")
                if sample_acc in OHtable:
                    vecs = line.strip("\n").split(" ")
                    temp_vecs.setdefault(sample_acc, [])
                    temp_vecs[sample_acc].append([float(vec) for vec in vecs])
                    
        duplicate_count = 0
        for sample_acc in temp_vecs:
            mean_vec = np.mean(temp_vecs[sample_acc], axis = 0)
            DataMat["acc"].append(sample_acc)
            DataMat["answer"].append(OHtable[sample_acc][0])
            for i in range(len(mean_vec)):
                DataMat["MBGD{}_CDHIT{}_v{}".format(clstrMBGD, clstrCDHIT, i)].append(mean_vec[i])
            if len(temp_vecs[sample_acc]) > 1: duplicate_count += 1
        #print(duplicate_count)
        
    DataMat_list.append(pd.DataFrame(DataMat))


# In[98]:


DataMat_list[0]


# In[76]:


DataMat_list[2]


# In[74]:


DataMat_list[2]["answer"].value_counts()["O8"]


# In[94]:


df1 = DataMat_list[0]
df2 = DataMat_list[2]

# acc 열을 기준으로 합치기 (inner join)
merged_df = pd.merge(df1, df2, on='acc', how='outer').drop(columns=['answer_x']).rename(columns={'answer_y': 'answer'})
columns_order = ['acc', 'answer'] + [col for col in merged_df.columns if col not in ['acc', 'answer']]
merged_df = merged_df[columns_order]

# `_v0`로 끝나는 열 찾기
for i in range(320):
    v_columns = [col for col in merged_df.columns if re.search(f'_v{i}$', col)]  # `_v{i}`로 끝나는 열 찾기
    if v_columns:  # 해당 그룹이 존재할 경우만 평균 계산
        merged_df[f'avg_v{i}'] = merged_df[v_columns].mean(axis=1, skipna=True)

# 새로운 DataFrame을 생성 (acc, answer, avg_v0~avg_v319 포함)
selected_columns = ['acc', 'answer'] + [col for col in merged_df.columns if re.match(r'avg_v\d+$', col)]
wzx_df = merged_df[selected_columns]

wzx_df


# In[110]:


Y = wzx_df["answer"].astype("category")
X = wzx_df.drop(["answer", "acc"], axis=1)
#Y = DataMat_list[0]["answer"].astype("category")
#X = DataMat_list[0].drop(["answer", "acc"], axis=1)

RF_estimator = RandomForestClassifier(n_estimators=100, random_state=int(time.time()), n_jobs=40, criterion="entropy", max_depth=10)#, class_weight="balanced")

"""#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/5, stratify=Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/5)
RF_estimator.fit(x_train, y_train)

y_pred = RF_estimator.predict(x_test)    
pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()"""

scores = cross_val_score(RF_estimator, X, Y, cv=StratifiedKFold(n_splits = 10))

print(scores.mean(), scores.std())

