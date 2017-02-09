"""
This module takes data as numpy arrays and perform fitting with a FC neural network model, 
consider merging into data preparation one?
"""

def precrec(x_set,y_set,model,modif=None,verbose=1):
    num=len(y_set[0])
    prec=[[] for i in range(num)]
    rec=[[] for i in range(num)]
    thres=[[] for i in range(num)]
    #auc=[0. for i in range(num)]
    yhat=model.predict(x_set,verbose)
    if modif is not None:
        yhat=modif(yhat)
    if len(yhat)==2:
        yhat=yhat[0]
    for i in range(num):
        prec[i],rec[i],thres[i]=np.array(precision_recall_curve(y_set[:,i],yhat[:,i]))
        #auc[i]=roc_auc_score(y_set[:,i],yhat[:,i])
        plt.plot(rec[i],prec[i])
    plt.show()
    return prec, rec, thres, yhat

def roccurveauc(x_set,y_set,model,verbose=1):
    num=len(y_set[0])
    fpr=[[] for i in range(num)]
    tpr=[[] for i in range(num)]
    thres=[[] for i in range(num)]
    auc=[0. for i in range(num)]
    try:
        yhat=model.predict(x_set,verbose)
    except:
        yhat=model.predict(x_set)
    if len(yhat)==2:
        yhat=yhat[0]
    for i in range(num):
        fpr[i],tpr[i],thres[i]=np.array(roc_curve(y_set[:,i],yhat[:,i]))
        auc[i]=roc_auc_score(y_set[:,i],yhat[:,i])
        plt.plot(fpr[i],tpr[i])
    plt.show()
    return auc, fpr, tpr, thres

def f1scoretuning(x,y,model,modif=None):
    prec,rec,thres=precrec(x,y,model,modif)[0:3]
    tune=[]
    maxf1=[]
    for pp,rr,tt in zip(prec,rec,thres):
        f1sc_arr=[]
        for ppj,rrj in zip(pp,rr):
            if (ppj+rrj)==0.:
                f1sc_arr+=[0.]
            else:
                f1sc_arr+=[2*ppj*rrj/(ppj+rrj)]
        tune+=[tt[np.argmax(f1sc_arr)]]
        maxf1+=[max(f1sc_arr)]
    return tune,maxf1

def jaccardindextuning(x,y,model,modif=None):
    thres,yhat=precrec(x,y,model,modif)[2:4]
    tune=[]
    maxji=[]
    for i,tt in enumerate(thres):
        ji_arr=[]
        for tti in tt:
            ji_arr+=[jaccard_index(y[:,i],np.ceil(yhat[:,i]-tti))]
        #print(ji_arr)
        tune+=[tt[np.argmax(ji_arr)]]
        maxji+=[max(ji_arr)]
    return tune,maxji

def jaccard_index(y,yhat):
    return len([yi for yi,yy in enumerate(y) if yy==1 and yhat[yi]==1])/len([yi for yi,yy in enumerate(y) if yy==1 or yhat[yi]==1])

def bayup(yhat):
    temp=[]
    for ii in yhat:
        qq=np.array([sum(pij[i]*ii) for i in range(numb_ing)])
        qq=qq*sum(list(ii))/sum(qq)
        temp+=[qq]
    return np.array(temp)

def bayuplin(yhat):
    temp=[]
    for ii in yhat:
        qq=np.array([sum(pij[i]*ii) for i in range(numb_ing)])
        temp+=[qq]
    return np.array(temp)

def visualtest(nn,predD,thres):
    yhat=predD
    retr=list(ret_test[nn])
    ingret=ingrototret[retr[0]]
    print('Recipe id, image path, and recipe name:\n',retr+[ingret[0]])
    print('"Truth":',ingret[1:] )
    if len(yhat)!=2:
        print('Predicted:', [ingdict[i] for i,ii in enumerate(yhat[nn]) if ii>thres[i]])
    else:
        print('Predicted:',[[catgdict[ii] for ii in np.argsort(yhat[1][nn])[-1:]][0],
                            [ingdict[i] for i,ii in enumerate(yhat[0][nn]) if ii>thres[i]]])
    plt.imshow(ndimage.imread(homedir+'/'+retr[1])[13:-13,13:-13,:])
    plt.show()
    return
    
def visualtest_man(img_name,model,thres):
    img_in=ndimage.imread(scriptscratchdir+'/'+img_name)
    img_out=modelvggc.predict(np.array([imresize(img_in,(224,224,3))]))
    yhat=model.predict(np.array(img_out))
    if len(yhat)!=2:
        print('Predicted:', [ingdict[i] for i,ii in enumerate(yhat[0]) if ii>thres[i]])
    else:
        print('Predicted:',[[catgdict[ii] for ii in np.argsort(yhat[1][0])[-1:]][0],
                            [ingdict[i] for i,ii in enumerate(yhat[0][0]) if ii>thres[i]]])
    plt.imshow(img_in)
    plt.show()
    return