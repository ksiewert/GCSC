import glob
import numpy as np
import pandas as pd
import pdb
import numpy.ma as ma
import sys
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.formula.api import wls
import math
from statsmodels.stats.anova import anova_lm
import scipy
import scipy.stats.mstats
import argparse as ap
import scipy.stats as stats
import pickle
import os
import os.path 

#This shouldnt' rely on 0/1 binary gene sets
def prepareDataCoexp(data,setMatrix,setColNames,corr2):
    for gSet in setColNames:
        data[gSet+"_annotValue"] = setMatrix[data["Gene_Name"]].loc[gSet].tolist() #will be 0/1 if binary gene set
        data[gSet] = getPathCoexp(data[gSet+"_annotValue"],corr2)

        
def prepareDataNoCoexp(data,setMatrix,setColNames):
    for gSet in setColNames:
        data[gSet]=setMatrix[data["Gene_Name"]].loc[gSet].tolist()
        data[gSet+"_annotValue"]=setMatrix[data["Gene_Name"]].loc[gSet].tolist()

#Will want to check for binary annotations before calculating
def calcEnrich(coef,setMembership,setName,freeInter):
    if freeInter:
        coef=coef[:-1]
    inSet = setMembership[setMembership[setName+"_annotValue"]==1]
    setH2 = inSet.multiply(np.array(coef)).values.sum()
    allH2 = setMembership.multiply(np.array(coef)).values.sum()
    M=setMembership.shape[0]
    n=inSet.shape[0]
    if allH2==0 or M==0 or n==0:
        return 1,allH2
    #if one gene set, equivalent to (n*(coef[0]+coef[1]))/(n*(coef[0]+coef[1])+(M-n)*coef[1])/(n/M)
    return (setH2/allH2)/(n/M),allH2

#Calculate h2(C)/|C|-)h^2-h^2(C)/(M-|C|)
def calcDiffProp(coef,setMembership,setName,freeInter):
    if freeInter:
        coef=coef[:-1]
    inSet = setMembership[setMembership[setName+"_annotValue"]==1]
    setH2 = inSet.multiply(np.array(coef)).values.sum()
    notSet = setMembership[setMembership[setName+"_annotValue"]==False]
    notsetH2 = notSet.multiply(np.array(coef)).values.sum()
    M=setMembership.shape[0]
    n=inSet.shape[0]
    notset=M-n
    if notset==0 or n==0:
        return 0
    return setH2/n-notsetH2/notset

def getTWASforGenes(TWAStrait,tissue,allGenes):
    TWASinfo = []
    ENSG = []
    hsq=[]
    r2=[]
    chrs=[]
    pos=[]
    dic = pickle.load(open("/n/groups/price/katie/UKB_TWAS_GTExv7models/Formatted_AllTissues_inchsq/OnlySigGenes/"+tissue+"_"+TWAStrait+"_TWASres.pkl",'rb'))

    for geneEntry in dic.keys():
        gene = geneEntry[1]
        if gene.split(".")[0] in allGenes:
            TWASinfo.append(float(dic[geneEntry][0]))
            hsq.append(float(dic[geneEntry][1]))
            ENSG.append(gene)
            r2.append(float(dic[geneEntry][3]))
            chrs.append(int(dic[geneEntry][5])) #Changed 3/2
            pos.append(int(dic[geneEntry][6]))
    return np.array(TWASinfo)**2,np.array(ENSG),np.array(hsq),np.array(chrs),np.array(pos),np.array(r2)


def getCoexpMatrix(tissue,allGenes):
    corr2 = np.load(open("/n/groups/price/katie/GenePredAnalysis/AllGeneCoexpMatrics/OnlySigGenes/coexpMatrix_"+tissue+".npy",'rb'))
    allModelGenes = np.load(open("/n/groups/price/katie/GenePredAnalysis/AllGeneCoexpMatrics/OnlySigGenes/geneNames_"+tissue+".npy",'rb'))
    allModelGenes = np.char.replace(allModelGenes,tissue+".","")
    allModelGenesENSG = [ENSG.split(".")[0] for ENSG in allModelGenes] 
    universeGenes,corri,allGenesi = np.intersect1d(allModelGenesENSG,allGenes,return_indices=True) #get genes in allGenes that are in coexpression matrix
    corr2 = corr2[np.ix_(corri,corri)]
    #corr2 = np.where(corr2<.1,0,corr2) #may want to delete
    return(corr2,universeGenes)


def getPathCoexp(gSet,corr2):
    '''
    Params:
    gSet:Vector of 0/1 abscense/prescence in the gene set
    corr2:Squared co-regulation score
    '''
    coexpMatrixSet = corr2*np.array(gSet) #Want to multiply along columns
    #pdb.set_trace()
    return(np.sum(coexpMatrixSet,axis=1))

    
def runGCSC(data,coef,contCoef,setNameFile,trait,freeIntercept):
    X = np.array(data[coef + contCoef])

    nGenesPerTiss=data.shape[0]/data["tissue"].nunique()
    hgeAll_hat = np.mean(data["Z2"]-1)*nGenesPerTiss/(Ntrait*np.mean(data["all"]))
    
    weights=data["genetissCountW"]*(1./data["all_count"])*1./((1+Ntrait*hgeAll_hat*data["all"]/nGenesPerTiss)**2)
    
    if freeIntercept:
        X = np.hstack((X,np.ones((X.shape[0],1))))
    XTw = X.T*np.array(weights)[None,:]
    XTX = XTw@X
    XTy = XTw@data["y"]

    sol=np.linalg.solve(XTX,XTy)
    print(sol)
    setstoUse = coef
    modelenriches = np.zeros(len(setstoUse))
    modeldiffprop = np.zeros(len(setstoUse))
    modeltaustar = np.zeros(len(setstoUse))

    for gseti in range(len(coef)):
        gSet = coef[gseti]
        #pdb.set_trace()
        modelenriches[gseti], h2g = calcEnrich(sol,data[[setName+"_annotValue" for setName in setstoUse]+contCoef],gSet,freeIntercept)
        modeldiffprop[gseti] = calcDiffProp(sol,data[[setName+"_annotValue" for setName in setstoUse]+contCoef],gSet,freeIntercept)
        sd_c = np.std(data[[gSet+"_annotValue"]])
        modeltaustar[gseti] = (sol[gseti])*sd_c/(h2g/data.shape[0])
    
    modelcovtaustar = np.zeros(len(contCoef))              
    for gseti in range(len(coef),len(coef)+len(contCoef)):
        gSet = contCoef[gseti-len(coef)]
        sd_c=np.std(data[gSet])
        modelcovtaustar[gseti-len(coef)] = (sol[gseti])*sd_c/(h2g/data.shape[0])
    blocks = data["block"].unique()
    nBlocks=len(blocks)
    coef_joint_set = np.zeros((len(setstoUse),nBlocks),dtype=np.float32)
    enriches = np.zeros((len(setstoUse),nBlocks),dtype=np.float32)
    diffprop = np.zeros((len(setstoUse),nBlocks),dtype=np.float32)
    ps_joint_set = np.zeros((len(setstoUse),nBlocks),dtype=np.float32)
    ps_enrich = np.zeros((len(setstoUse),nBlocks),dtype=np.float32)
    ps_diff = np.zeros((len(setstoUse),nBlocks),dtype=np.float32)
    coef_covar = np.zeros((len(contCoef),nBlocks),dtype=np.float32)
    covar_tarstar=np.zeros((len(contCoef),nBlocks),dtype=np.float32)
    ps_covartarstar=np.zeros((len(contCoef),nBlocks),dtype=np.float32)
    if freeIntercept:
        inters = np.zeros((nBlocks),dtype=np.float32)
        ps_inters = np.zeros((nBlocks),dtype=np.float32)
        
    mjs = np.zeros((nBlocks),dtype=np.float32)
    hjs = np.zeros((nBlocks),dtype=np.float32)
    totalweights=np.sum(data["genetissCountW"]*(1./data["all_count"])*1./((1+Ntrait*hgeAll_hat*data["all"]/nGenesPerTiss)**2))
    for blocki in range(len(blocks)):
        currRows = data.loc[data.block!=blocks[blocki]]
        currBlock = data.loc[data.block==blocks[blocki]]
        X_block=np.array(currBlock[coef + contCoef])
        if freeIntercept:
            X_block = np.hstack((X_block,np.ones((X_block.shape[0],1))))
        weights_block=currBlock["genetissCountW"]*(1./currBlock["all_count"])*1./((1+Ntrait*hgeAll_hat*currBlock["all"]/nGenesPerTiss)**2) 

        XTX_block = X_block.T@np.diag(weights_block)@X_block
        XTy_block = X_block.T@np.diag(weights_block)@currBlock["y"]
        sol_block=np.linalg.solve(XTX-XTX_block,XTy-XTy_block)
        mj = np.sum(weights_block) 

        hj = totalweights / mj
        mjs[blocki] = mj
        hjs[blocki] = hj
        for geneseti in range(len(setstoUse)):    
            gSet = setstoUse[geneseti]
            coef_joint_set[geneseti,blocki] = sol_block[geneseti]
            enriches[geneseti,blocki],h2g = calcEnrich(sol_block,currRows[[setName+"_annotValue" for setName in setstoUse]+contCoef],gSet,freeIntercept)
            diffprop[geneseti,blocki] = calcDiffProp(sol_block,currRows[[setName+"_annotValue" for setName in setstoUse]+contCoef],gSet,freeIntercept)

            ps_joint_set[geneseti,blocki] = hj*sol[geneseti]-(hj-1)*coef_joint_set[geneseti,blocki]
            ps_enrich[geneseti,blocki] = hj*modelenriches[geneseti]-(hj-1)*enriches[geneseti,blocki]
            ps_diff[geneseti,blocki] = hj*modeldiffprop[geneseti]-(hj-1)*diffprop[geneseti,blocki]
        if freeIntercept:
            inters[blocki]=sol_block[-1]
            ps_inters =  hj*sol[-1]-(hj-1)*sol_block[-1]
        for gseti in range(len(coef),len(coef)+len(contCoef)):
            gSet = contCoef[gseti-len(coef)]
            coef_covar[gseti-len(coef),blocki]=sol_block[gseti]

    if not os.path.exists(args.outDir+"/"+setNameFile):
        os.mkdir(args.outDir+"/"+setNameFile)

    out = open(args.outDir+"/"+setNameFile+"/"+setNameFile+"_"+trait+".txt",'w')

    for geneseti in range(len(setstoUse)):
        gSet = setstoUse[geneseti]
        nGenesModelPairs = data[data[gSet+"_annotValue"]==1].shape[0]

        nGenes = data[data[gSet+"_annotValue"]==1]["Gene_Name"].nunique()
        thetaJ_joint_set= nBlocks*sol[geneseti]-np.sum(((totalweights-mjs)*np.array(coef_joint_set[geneseti]))/totalweights)
        thetaJ_enrich = nBlocks*modelenriches[geneseti]-np.sum(((totalweights-mjs)*np.array(enriches[geneseti]))/totalweights)
        thetaJ_diff = nBlocks*modeldiffprop[geneseti]-np.sum(((totalweights-mjs)*np.array(diffprop[geneseti]))/totalweights)
        sd_c = np.std(data[gSet+"_annotValue"])
        nBlocks=data["block"].nunique()
        taustarJ = thetaJ_joint_set*sd_c/(h2g/data.shape[0])
        
        se_joint_set = np.sqrt(1/nBlocks*np.sum(np.square(ps_joint_set[geneseti]-thetaJ_joint_set)/(hjs-1)))
        se_diff = np.sqrt(1/nBlocks*np.sum(np.square(ps_diff[geneseti]-thetaJ_diff)/(hjs-1)))
        se_enrich = np.sqrt(1/nBlocks*np.sum(np.square(ps_enrich[geneseti]-thetaJ_enrich)/(hjs-1)))
        se_tau_star = se_joint_set*sd_c/(h2g/data.shape[0])
        pdb.set_trace()
        out.write(setstoUse[geneseti]+"_coef: "+str(thetaJ_joint_set)+" "+str(se_joint_set)+" "+str(scipy.stats.t.sf(abs(thetaJ_joint_set/se_joint_set),nBlocks)*2.)+"\n")
        
        if se_tau_star>0:
            out.write(setstoUse[geneseti]+"_tau*: " + str(taustarJ) + " " + str(se_tau_star)+" "+str(scipy.stats.t.sf(abs(taustarJ/se_tau_star),nBlocks)*2.) +"\n")
            out.write(setstoUse[geneseti]+"_enrichment: "+str(thetaJ_enrich)+" "+str(se_enrich)+" "+str(scipy.stats.t.sf(abs(thetaJ_diff/se_diff),nBlocks)*2.)+"\n")

        else:
            out.write(setstoUse[geneseti]+"_tau*: " + str(taustarJ) + " " + str(se_tau_star)+" 0"+"\n")
            out.write(setstoUse[geneseti]+"__enrichment: "+str(thetaJ_enrich)+" "+str(se_enrich)+" 0"+"\n")
        out.write(setstoUse[geneseti]+"_genecounts: "+str(nGenesModelPairs)+" "+str(nGenes)+"\n")
    if freeIntercept:
        thetaJ_inter = nBlocks*sol[-1]-np.sum(((totalweights-mjs)*np.array(inters))/totalweights)
        se_inter = np.sqrt(1/nBlocks*np.sum(np.square(ps_inters-thetaJ_inter)/(hjs-1)))
        out.write("Intercept:" + str(thetaJ_inter)+" "+str(se_inter) +"\n")
        
    for geneseti in range(len(coef),len(coef)+len(contCoef)):
        gSet = contCoef[geneseti-len(coef)]
        theta_covar=nBlocks*sol[geneseti]-np.sum(((totalweights-mjs)*np.array(coef_covar[geneseti-len(coef)]))/totalweights)
        se_covar=np.sqrt(1/nBlocks*np.sum(np.square(ps_covartarstar[geneseti-len(coef)]-theta_covar)/(hjs-1)))

       
        out.write(gSet+"_coef: "+str(theta_covar)+" "+str(se_joint_set)+ str(se_tau_star)+str(scipy.stats.t.sf(abs(theta_covar/se_joint_set),nBlocks)*2.)+"\n")
        sd_c = np.std(data[gSet+"_annotValue"])
        taustar = theta_covar*sd_c/(h2g/data.shape[0])
        se_tau_star = se_covar*sd_c/(h2g/data.shape[0])
        out.write(gSet+"_tau*: " + str(taustarJ) + " " + str(se_tau_star)+str(scipy.stats.t.sf(abs(taustarJ/se_tau_star),nBlocks)*2.) +"\n")
        out.write(gSet+"_sumvalue: "+str(np.sum(data[gSet+"_annotValue"]))+"\n")

    out.close()
    
    
    
argp = ap.ArgumentParser(description="Simulate TWAS using real genotype data",
                         formatter_class=ap.ArgumentDefaultsHelpFormatter)
argp.add_argument("--geneSets",type=str)
argp.add_argument("--setInfo", type=str)
argp.add_argument("--outDir", type=str)
argp.add_argument("--trait", type=str)
argp.add_argument("--tissueFile", type=str)
argp.add_argument("--freeInter", default=False,action='store_true', help="Allow the intercept to vary. For use in estimating all gene heritability instead of gene set enrichment")

args = argp.parse_args()


#Get list of all genes in any of the sets
trait = args.trait

if args.tissueFile==None:
    tissList = pd.read_csv(open("/n/groups/price/katie/UKB_TWAS_GTExv7models/TWAS_Results_all13000genes/AllTissues/AllTiss.txt",'r'),header=None)
    tissues = tissList.iloc[:, 0].tolist()
else:
    tissList=pd.read_csv(open(args.tissueFile),header=None)
    tissues = tissList.iloc[:, 0].tolist()



Ndic = {}
for line in open("/n/groups/price/katie/Data/AllGWAS_N.txt",'r'):
    Ndic[line.split("\t")[0]]=int(line.split("\t")[1])
Ntrait = Ndic[trait] 



#Get list of set names that are covariates versus gene sets
#Three column file: gene name, whether to use as covariate(with all gene sets) and whether to use raw value, or co-expression score with set in regression
setInfo=pd.read_csv(open(args.setInfo,'r'),names=["CovarSet","CoexpSet"],sep=" ")
covarNames=setInfo.index[setInfo["CovarSet"]==1].tolist()
setNames=setInfo.index[setInfo["CovarSet"]==0].tolist()
coexpNames=setInfo.index[setInfo["CoexpSet"]==1].tolist()
noCoexpNames=setInfo.index[setInfo["CoexpSet"]==0].tolist()

setMembership=pd.read_csv(open(args.geneSets,'r'),header=0,index_col=0)
sets=setInfo.index.tolist()
setMembership=setMembership[setMembership.index.isin(sets)]
discrete = setMembership.isin([0,1]).round(decimals=6).all(axis=1)
contNames = discrete[~ discrete].index
binaryNames = discrete[discrete].index


allTissData = pd.DataFrame()
for tissue in tissues:
#for tissue in ["Whole_Blood"]:    
    print(tissue)
    TWASscores,ENSG,hsq,chr,pos,r2 = getTWASforGenes(args.trait, tissue,setMembership.columns)
    ENSG = [ENSG.split(".")[0] for ENSG in ENSG]
    corr2,geneNames = getCoexpMatrix(tissue,ENSG)
    twasi = np.intersect1d(ENSG,geneNames,return_indices=True)[1]
    uncorr_coex=np.sum(corr2,axis=1)
    d = np.diag_indices_from(corr2)
    ratio = np.minimum(np.maximum(0,r2[twasi]),hsq[twasi])/hsq[twasi]
    meanratio=np.mean(ratio)
    corr2[d]= np.maximum(ratio,meanratio)
    data = pd.DataFrame({"Gene_Name":geneNames,'tissue':tissue,'Z2':TWASscores[twasi],'hsq':hsq[twasi],'all':np.sum(corr2,axis=1),'all_count':uncorr_coex,'chr':chr[twasi],"pos0":pos[twasi],"r2":r2[twasi]})

    if len(coexpNames)>0:
        prepareDataCoexp(data,setMembership,coexpNames,corr2)


    if len(noCoexpNames)>0:
        prepareDataNoCoexp(data,setMembership,noCoexpNames)
        
    data = data[data["Z2"]<max(80,0.001*Ntrait)] 
    #Do corrected co-reg
    
    allTissData = allTissData.append(data)
data=allTissData.sort_values(by=["chr","pos0"])
numBlocks=200
data["Universe"]=data["all"]
#Get list of genes, sorted by pos, then break into chunks
data=data.reset_index(drop=True)
data["block"]=data.index//(data.shape[0]/numBlocks)
data["block"]=data["block"].astype(int)
data.dropna(inplace=True,subset=["block"])
data["genetissCountW"] = 1/data["Gene_Name"].map(data["Gene_Name"].value_counts())

columns= coexpNames

groups=data.groupby('tissue')
means = groups["Universe"].mean()
topTiss=means.idxmax()
topMean = groups.get_group(topTiss)["Universe"].mean()
topStd = groups.get_group(topTiss)["Universe"].std()


for tissue,group in data.groupby('tissue'):
    tissMean=group["Universe"].mean()
    tissStd=group["Universe"].std()
    for column in columns:
        cVal = group[column]
        pSet=np.sum(group[column])/np.sum(group["Universe"])

#         data.loc[group.index,column]=(cVal/(tissStd/topStd)+(pSet*topMean-(cVal.mean()/(tissStd/topStd)).mean()))
        data.loc[group.index,column]=cVal/(tissStd/topStd)+(pSet*topMean-(cVal.mean()/(tissStd/topStd)))        

# print(data.groupby('tissue').mean()[["all","Universe","Olfactory"]])
# print(data.groupby('tissue').std()[["all","Universe","Olfactory"]])
# data.to_csv("/n/groups/price/katie/GenePredAnalysis/Draft1_freeinter/Allgenes_uncorrcoreg/Dataforvis/CoregStd_"+trait+".csv")
# exit() #WILL WANT TO REMOVE THIS AND TOW OTHER DATS>TOCSV LINES
data.to_csv(open("/n/groups/price/katie/GenePredAnalysis/Draft1_freeinter/Test_minr2hsqstrat/Data/"+trait+".csv",'w'))
data["y"]=(data["Z2"]-1)/Ntrait
if args.tissueFile==None:
    tissstr=""
else:
    tissstr=args.tissueFile.split("/")[-1].replace(".txt","")
if len(setNames)>0:
    for setName in setNames:
        coef = [setName] + covarNames
        binaryCoef=[i for i in coef if i in binaryNames]
        contCoef=[i for i in coef if i not in binaryNames]
        runGCSC(data,binaryCoef,contCoef,setName+tissstr,args.trait,args.freeInter)

else:   
    binaryCoef=[i for i in covarNames if i in binaryNames]
    contCoef=[i for i in covarNames if i not in binaryNames]
    runGCSC(data,binaryCoef,contCoef,"AllCovar"+tissstr,args.trait,args.freeInter)
