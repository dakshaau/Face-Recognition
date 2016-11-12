import sys
import os
import cv2, time
import numpy as np
import pickle
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_default.xml')

def browseAndSaveFaces(base):
    # images = []
    imfile = open(base+'\images.txt','w')
    lblfile = open(base+'\labels.txt','w')
    labels = []
    for i, (dirname, dirnames, filenames) in enumerate(os.walk(base)):
        if dirname == base:
            labels = [d for d in dirnames]
        count = 0
        for file in filenames:
            if base == 'probe':
                if file.endswith('.jpg') and not file.startswith('face'):
                    img = cv2.imread(os.path.join(dirname,file))
                    if saveFace(img,dirname,file):
                        imfile.write(os.path.join(dirname,'face_'+file)+'\n')
                        lblfile.write(labels[i-1]+'\n')
                        print('face_'+file)
                        break
            else:
                if file.endswith('.jpg') and not file.startswith('face'):
                    img = cv2.imread(os.path.join(dirname,file))
                    if saveFace(img,dirname,file):
                        imfile.write(os.path.join(dirname,'face_'+file)+'\n')
                        lblfile.write(labels[i-1]+'\n')
                        print('face_'+file)
                        count += 1
                        # if count == 4:
                        #     break
    imfile.close()
    lblfile.close()

def get_xy(rects):
    for x1, y1, x2, y2 in rects:
        xy = [x1,x2,y1,y2]
        return xy

def detect(img, cascade=face_cascade):
    img = cv2.equalizeHist(img)
    rects = cascade.detectMultiScale(img,minSize=(200, 200),flags=1,scaleFactor=1.2)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def correct_gamma(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def process_image(imag):  
    gr = correct_gamma(imag, 0.3) 
    gr = cv2.equalizeHist(gr)
    return gr

def saveFace(img,path='_',imname='_'):
    img_color = img #cv2.imread('image.jpg')
    h,w = img_color.shape[:2]
    ratio = w/float(h)
    img_color = cv2.resize(img_color, (int(ratio*720),720))     #frame from camera
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    faces = detect(img_gray)
    if len(faces) == 0:
        return False
    for rect in faces:
        xy1=get_xy(faces)
        if xy1 :
            fac = 30
            const = 6
            const2 = int(const/2)
            cropi=img_gray[xy1[2]+fac+const:xy1[3]-fac , xy1[0]+fac+const2:xy1[1]-fac-const2]
            cropp = cv2.resize(cropi, (100,100))
            cropp = process_image(cropp)
            fname = 'face_'
            cv2.imwrite(path+'/'+fname+imname,cropp)
    return True

def generateFeatures(path = 'gallery/'):
    surf = cv2.xfeatures2d.SURF_create(500)
    imgs = open(path+'images.txt','r').readlines()
    # labels = open(path+'labels.txt','r').readlines()
    imgs = [im.strip('\n') for im in imgs]
    # labels = [lbl.strip('\n') for lbl in labels]
    features = []
    print(path)
    x = float(len(imgs))
    for i, im in enumerate(imgs):
        img = cv2.imread(im,0)
        features.append(surf.detectAndCompute(img,None))
        print('\rCompleted: {:.2f}%'.format(((i+1)/x)*100),end = ' ')
    print()
    return features

def compareFeat(gallery, probe):
    bf = cv2.BFMatcher()
    g = len(gallery)
    p = len(probe)
    tot = float(g*p)
    mat = np.ndarray((g,p),np.float32)
    for i, (gk, gdes) in enumerate(gallery):
        for j, (pk, pdes) in enumerate(probe):
            matches = bf.match(gdes,pdes)
            matches = sorted(matches, key = lambda x:x.distance)
            # np.nanmean(matches[:100],dtype=np.float32)
            match = [m.distance for m in matches[:500]]
            mat[i,j] = np.mean(match,dtype=np.float32)
            # mat[i,j] = matches[0].distance
            print('\rCompleted: {:.2f}%'.format(((((i*p)+(j+1))/tot)*100)), end = ' ')
    print()
    return mat

def getLabelMat(gpath, ppath):
    g = open(gpath+'labels.txt','r').readlines()
    glabs = [i.strip('\n') for i in g]
    p = open(ppath+'labels.txt','r').readlines()
    plabs = [i.strip('\n') for i in p]
    g = len(glabs)
    p = len(plabs)
    mat = np.ndarray((g,p),dtype=np.bool_)
    for i, glab in enumerate(glabs):
        for j, plab in enumerate(plabs):
            mat[i,j] = glab == plab
    return mat

def getGenuineImposter(comp, match):
    genuine = []
    imposter = []
    r,c = np.where(match == True)
    # for i in range(r.size):
    #     genuine.append(comp[r[i],c[i]])
    C = np.unique(c)
    for i in range(C.size):
        ind = np.where(c == C[i])
        # print(ind[0])
        genuine.append(comp[r[ind[0][:]],C[i]].min())
    print(len(genuine),min(genuine),max(genuine))
    r,c = np.where(match == False)
    # print(r,c)
    for i in range(r.size):
        imposter.append(comp[r[i],c[i]])
    # C=np.unique(r)
    # for i in range(C.size):
    #     ind = np.where(r == C[i])
    #     # print(ind[0])
    #     imposter.append(np.mean(comp[C[i],c[ind[0][:]]]))
    print(len(imposter),min(imposter),max(imposter))
    # pass
    return genuine, imposter

def drawGI(genuine, imposter):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(0.0,1.0,num = 101)
    # print(x)
    yg = []
    yi = []
    for i in range(x.size):
        r = np.where(genuine <= x[i])
        if i > 0:
            yg.append(r[0].size - sum(yg))
        else:
            yg.append(r[0].size)
        r = np.where(imposter <= x[i])
        if i > 0:
            yi.append(r[0].size - sum(yi))
        else:
            yi.append(r[0].size)
    # print(yg,yi)
    y1 = [float(i)/sum(yg) for i in yg]
    # bins = len(set(genuine))
    # print(bins)
    plt.plot(x,y1,color = 'green',label='Genuine')
    y2 = [float(i)/sum(yi) for i in yi]
    plt.plot(x,y2,color = 'red',label='Imposter')
    plt.axvline(x=0.3849,linestyle='dashed',color='black',label='Threshold = 0.385')
    plt.xlabel('Face Match Scores')
    plt.ylabel('Relative Frequency')
    plt.title('Genuine and Imposter distributions')
    plt.xlim(0.0, 1.0)
    plt.legend()
    fig.savefig('GenuineImposter.png')
    plt.show()

def plotROC(genuine, imposter):
    fig = plt.figure()
    rng = np.linspace(0,1.0,num=101)
    tot_g = len(genuine)
    tot_i = len(imposter)
    TPR = []
    FPR = []
    for i in range(101):
        r = np.where(genuine <= rng[i])
        TPR.append(r[0].size/float(tot_g))
        r = np.where(imposter <= rng[i])
        FPR.append(r[0].size/float(tot_i))
    
    plt.plot(FPR,TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    # plt.axes([0.0,1.0,0.0,1.0])
    # fig.savefig('ROCPlot.png')
    plt.show()

def plotCMC(mat, matches, gpath='gallery/'):
    labels = open(gpath+'labels.txt','r').readlines()
    labels = [lbl.strip() for lbl in labels]
    labarr = np.asarray(labels)
    # print(labarr.shape)
    R = np.unique(labarr).tolist()
    cli = len(R)
    fig = plt.figure()
    ma = None
    match = None
    r,c = np.where(matches == True)
    C = np.unique(c).tolist()
    que = len(C)
    ma = np.ndarray((cli,que),dtype=np.float32)
    match = np.ndarray((cli,que),dtype=np.bool_)
    
    for i,row in enumerate(R):
        ind_r = np.where(labarr == row)
        for j,col in enumerate(C):
            # ind_c = np.where(c == col)

            temp = mat[ind_r[0],col]
            temp_m = matches[ind_r[0],col]

            idx = np.argsort(temp,axis=0)
            ma[i,j] = temp[idx[0]]
            match[i,j] = temp_m[idx[0]]

    ind = np.argsort(ma, axis=0)
    # print(ind)
    r,c = ind.shape
    x = list(range(1,r+1))
    y = []
    for i in range(r):
        count = 0.0
        for j in C:
            comp = ma[ind[:i,j],j]
            m = match[ind[:i,j],j]
            z = np.where(m == True)
            if z[0].size > 0:
                count += 1
        y.append(count/que)
    plt.plot(x,y)
    plt.xlabel('Rank Counted as Recognition')
    plt.ylabel('Recognition Rate')
    plt.title('CMC Curve')
    plt.xlim(1,r)
    fig.savefig('CMCCurve.png')
    plt.show()

if __name__ == "__main__":
    paths = ['gallery','probe']
    [browseAndSaveFaces(p) for p in paths]
    
    print('Generating features ...')
    galfeat = generateFeatures('gallery/')
    prfeat = generateFeatures('probe/')
    print('Done!')
    mat = None
    if not os.path.exists('dists.mat'):
        print('Generating Comparison Matrix ...')
        mat = compareFeat(galfeat,prfeat)
        pickle.dump(mat,open('dists.mat','wb'))
    else:
        print('Loading Comparison Matrix ...')
        mat = pickle.load(open('dists.mat','rb'))
        print('Done!')
    match = None
    if not os.path.exists('match.mat'):
        match = getLabelMat('gallery/','probe/')
        pickle.dump(match,open('match.mat','wb'))
    else:
        match = pickle.load(open('match.mat','rb'))
    # print(match.shape)
    # print(mat.shape)

    genuine, imposter = getGenuineImposter(mat, match)
    
    drawGI(genuine, imposter)
    plotROC(genuine, imposter)
    plotCMC(mat,match)
    # print(mat.max(),mat.min())
