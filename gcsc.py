import argparse as ap
import glob
import pandas as pd
import scipy.sparse
import numpy as np
import scipy.stats
import scipy.sparse as sparse

#Calculate h2(C)/|C|-)h^2-h^2(C)/(M-|C|)
def calcDiffProp(coef,setMembership,setName):
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

def calcEnrich(coef,setMembership,setName):
    coef = coef[:-1] #remove intercept
    isBinary = setMembership.isin([0,1]).all().all()
    allH2 = setMembership.multiply(np.array(coef)).values.sum()
    if isBinary:
        inSet = setMembership[setMembership[setName+"_annotValue"]==1]
        setH2 = inSet.multiply(np.array(coef)).values.sum()
        M = setMembership.shape[0]
        n = inSet.shape[0]
        if allH2==0 or M==0 or n==0:
            return 1,allH2
        return (setH2/allH2)/(n/M),allH2
    return None,allH2


def runGCSC(data,coefs,N,out):
    
    out=open(args.out+"/GCSCresults.txt",'w')
    out.write("Parameter Value Standard_error P-value\n")
    for coefi in range(len(coefs)):
        coef = coefs[coefi]
        currCol=["all"]+[coef]
        nGenesPerTiss=data.shape[0]/data["tissue"].nunique()
        hgeAll_hat = np.mean(data["Z2"]-1)*nGenesPerTiss/(N*np.mean(data["all_unstd"]))
        
        weights=data["genetissCountW"]*(1./data["All_nocorr"])*1./((1+N*hgeAll_hat*data["all_unstd"]/nGenesPerTiss)**2)
        totalweights=np.sum(weights)
        sol,XTX,XTy = regress(data[currCol],data["y"],weights)

        modelenriches, h2g = calcEnrich(sol,data[["all_annotValue"]+[coef+"_annotValue" ]],coef)
        if modelenriches!=None:
            modeldiffprop = calcDiffProp(sol,data[["all_annotValue"]+[coef+"_annotValue"]],coef)
        else:
            modeldiffprop=None
        sd_c = np.std(data[[coef+"_annotValue"]])
        taustar = (sol[1])*sd_c/(h2g/data.shape[0])
        
        #Now, need to jackknife
        nBlocks=data["block"].nunique()
        coef_joint_set = np.zeros(nBlocks,dtype=np.float32)
        enriches = np.zeros(nBlocks,dtype=np.float32)
        diffprop = np.zeros(nBlocks,dtype=np.float32)
        inters = np.zeros(nBlocks,dtype=np.float32)
        ps_joint_set = np.zeros(nBlocks,dtype=np.float32)
        ps_enrich = np.zeros(nBlocks,dtype=np.float32)
        ps_diff = np.zeros(nBlocks,dtype=np.float32)
        ps_inters = np.zeros(nBlocks,dtype=np.float32)
        coef_covar = np.zeros(nBlocks,dtype=np.float32)
        mjs = np.zeros((nBlocks),dtype=np.float32)
        hjs = np.zeros((nBlocks),dtype=np.float32)
        for blocki in range(nBlocks):
            coef_joint_set[blocki],enriches[blocki], diffprop[blocki], ps_joint_set[blocki], ps_enrich[blocki], ps_diff[blocki], inters[blocki], ps_inters[blocki], mjs[blocki], hjs[blocki] = jackknife(data,blocki,XTX,XTy,coef,N,hgeAll_hat,nGenesPerTiss,currCol,modelenriches,modeldiffprop,sol,totalweights)
         
        
        #Write output
        thetaJ_joint_set= nBlocks*sol[1]-np.sum(((totalweights-mjs)*np.array(coef_joint_set))/totalweights)
        if modelenriches!=None:
            thetaJ_enrich = nBlocks*modelenriches-np.sum(((totalweights-mjs)*np.array(enriches))/totalweights)
            thetaJ_diff = nBlocks*modeldiffprop-np.sum(((totalweights-mjs)*np.array(diffprop))/totalweights)
        thetaJ_inter = nBlocks*sol[-1]-np.sum(((totalweights-mjs)*np.array(inters))/totalweights)

        sd_c = np.std(data[coef+"_annotValue"])
        taustarJ = thetaJ_joint_set*sd_c/(h2g/data.shape[0])
        se_joint_set = np.sqrt(1/nBlocks*np.sum(np.square(ps_joint_set-thetaJ_joint_set)/(hjs-1)))
        if modelenriches!=None:
            se_diff = np.sqrt(1/nBlocks*np.sum(np.square(ps_diff-thetaJ_diff)/(hjs-1)))
            se_enrich = np.sqrt(1/nBlocks*np.sum(np.square(ps_enrich-thetaJ_enrich)/(hjs-1)))
        se_tau_star = se_joint_set*sd_c/(h2g/data.shape[0])
        se_inter = np.sqrt(1/nBlocks*np.sum(np.square(ps_inters-thetaJ_inter)/(hjs-1)))
        
        out.write(coef+"_coef: "+'{:0.3e}'.format(thetaJ_joint_set)+" "+'{:0.3e}'.format(se_joint_set)+" "+'{:0.3e}'.format(scipy.stats.t.sf(abs(thetaJ_joint_set/se_joint_set),nBlocks)*2.)+"\n")
        out.write(coef+"_tau*: " + '{:0.3e}'.format(taustarJ) + " " + '{:0.3e}'.format(se_tau_star)+" "+'{:0.3e}'.format(scipy.stats.t.sf(abs(taustarJ/se_tau_star),nBlocks)*2.) +"\n")
        if modelenriches!=None:
            out.write(coef+"_enrichment: "+str(round(thetaJ_enrich,3))+" "+'{:0.3e}'.format(se_enrich)+" "+'{:0.3e}'.format(scipy.stats.t.sf(abs(thetaJ_diff/se_diff),nBlocks)*2.)+"\n")

        out.write(coef+"_intercept: " + '{:0.3e}'.format(thetaJ_inter)+" "+'{:0.3e}'.format(thetaJ_inter) +"\n")
        
        
def jackknife(data,blocki,XTX,XTy,coef,N,hgeAll_hat,nGenesPerTiss,currCols,modelenriches,modeldiffprop,sol,totalweights):
    
    currBlock = data.query("block == @blocki")
    currRows = data.query("block != @blocki")
    X_block = np.hstack((currBlock[currCols],np.ones((currBlock[currCols].shape[0],1))))
    weights_block = currBlock["genetissCountW"]*(1./currBlock["All_nocorr"])*1./((1+N*hgeAll_hat*currBlock["all_unstd"]/nGenesPerTiss)**2)
    XTX_block = X_block.T@np.diag(weights_block)@X_block
    XTy_block = X_block.T@np.diag(weights_block)@currBlock["y"]
    sol_block=np.linalg.solve(XTX-XTX_block,XTy-XTy_block)
    mj = np.sum(weights_block)
    hj = totalweights / mj
    ps_joint_set = hj*sol[1]-(hj-1)*sol_block[1]
    
    inters = sol_block[-1]
    ps_inters =  hj*sol[-1]-(hj-1)*sol_block[-1]
    if modelenriches!=None:
        enrich ,_ = calcEnrich(sol_block,currRows[["all_annotValue"]+[coef+"_annotValue"]],coef)
        diffprop = calcDiffProp(sol_block,currRows[["all_annotValue"]+[coef+"_annotValue"]],coef)
        ps_enrich = hj*modelenriches-(hj-1)*enrich
        ps_diff = hj*modeldiffprop-(hj-1)*diffprop
        return sol_block[1],enrich, diffprop,ps_joint_set,ps_enrich,ps_diff,inters,ps_inters, mj, hj
    return sol_block[1], None, None,ps_joint_set,None,None,inters,ps_inters, mj, hj

    
    
    
def regress(X,y,weights):
    X = np.hstack((X,np.ones((X.shape[0],1)))) #Adds intercept
    XTw = X.T*np.array(weights)[None,:]
    XTX = XTw@X
    XTy = XTw@y
    sol=np.linalg.solve(XTX,XTy)
    return sol,XTX,XTy
    
def getTWASStats(path,tissue):
    '''Load in FUSION formatted statistics'''
    
    df = pd.concat(map(lambda file: pd.read_csv(file, usecols=["FILE","CHR","P0","TWAS.Z"],dtype={"FILE":str,"CHR":int,"P0":int,"TWAS.Z":str},sep="\t"), glob.glob(path.replace("tissue",tissue)+"/*.dat")))
    df = df[~df['TWAS.Z'].str.contains("NA")]
    df["ENSG"]=df["FILE"].str.split("/").str[-1].str.split(".").str[1]
    df["Z2"]=df["TWAS.Z"].astype(float)**2

    return df



def prepareDataCoexp(data,setMatrix,corr2):
    for gSet in setMatrix.index:
        data[gSet+"_annotValue"] = setMatrix[data["Gene"]].loc[gSet].tolist()
        
    coreg = sparse.csr_matrix.dot(corr2, data[[gSet+"_annotValue" for gSet in setMatrix.index]])
    for gSeti in range(len(setMatrix.index)):
        data[setMatrix.index[gSeti]] = sparse.csr_matrix.dot(corr2, data[[gSet+"_annotValue" for gSet in setMatrix.index]])[:,gSeti] 

        
        
argp = ap.ArgumentParser(description="Run GCSC regression", formatter_class=ap.ArgumentDefaultsHelpFormatter)
argp.add_argument("--geneSets",type=str,required=True)
argp.add_argument("--out", type=str)
argp.add_argument("--TWASdir", type=str,required=True,help="Directory with TWAS results in")
argp.add_argument("--N", type=int, help="GWAS sample size",required=True)
argp.add_argument("--coreg", type=str,required=True, help="Directory containing coregulation scores")
argp.add_argument("--tissues", type=str,help="Space seperated list of tissues to use, if not using all",nargs='*',action='store')
args = argp.parse_args()


if args.tissues!=None:
    tissues=args.tissues
else:
    tissues=[i.split("/")[-1].replace("_coregscores.npz","") for i in glob.glob(args.coreg+"/*"+"coregscores.npz")]

setMembership=pd.read_csv(open(args.geneSets,'r'),header=0,index_col=0)
sets=setMembership.index.tolist()
allTissData = pd.DataFrame()
for tissue in sorted(tissues):
    print(tissue)
    stats = getTWASStats(args.TWASdir, tissue)
    corrscores = scipy.sparse.load_npz(args.coreg+"/"+tissue+"_coregscores.npz")
    geneNames = np.loadtxt(args.coreg+"/"+tissue+"_geneNames.txt",dtype=str)
    

    data = pd.DataFrame({"Gene":geneNames,"all":corrscores.sum(axis=0).tolist()[0]})
    

    if len(sets)>0:
        prepareDataCoexp(data,setMembership,corrscores)
    corrscores.setdiag(1, k=0)
    data["All_nocorr"] = corrscores.sum(axis=0).tolist()[0]
    data=data.merge(stats,left_on="Gene",right_on="ENSG")
    data["tissue"]=tissue
    allTissData = allTissData.append(data)
    
    
allTissData = allTissData[allTissData["Z2"]<max(80,0.001*args.N)] 
numBlocks=200
allTissData=allTissData.sort_values(by=["CHR","P0"])
allTissData=allTissData.reset_index(drop=True)
allTissData["block"]=allTissData.index//(allTissData.shape[0]/numBlocks)
allTissData["block"]=allTissData["block"].astype(int)

#Standardize co-regulation scores
groups=allTissData.groupby('tissue')
allTissData["all_unstd"]=allTissData["all"]
means = groups["all"].mean()
topTiss=means.idxmax()
topMean = groups.get_group(topTiss)["all"].mean()
topStd = groups.get_group(topTiss)["all"].std()


for tissue,group in groups:
    tissMean=group["all"].mean()
    tissStd=group["all"].std()
    for column in sets +  ["all"]:
        cVal = group[column]
        pSet=np.sum(group[column])/np.sum(group["all"])
        allTissData.loc[group.index,column]=cVal/(tissStd/topStd)+(pSet*topMean-(cVal.mean()/(tissStd/topStd)))


allTissData["y"]=(allTissData["Z2"]-1)/args.N
allTissData["all_annotValue"]=1
allTissData["genetissCountW"] = 1/allTissData["Gene"].map(allTissData["Gene"].value_counts())
runGCSC(allTissData,sets,args.N,args.out)
