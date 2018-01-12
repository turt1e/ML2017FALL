import numpy as np
import sys
img_path=sys.argv[1]
tag_path=sys.argv[2]
print('img=',img_path)
print('targ=',tag_path)
import os
from skimage import io


fpath=img_path
#print(os.listdir(fpath))
#=========

def meanpic(pics):
    print('type=',type(pics[0]))
    print('len=',len(pics))
    img=np.zeros((pics[0].shape[0],pics[0].shape[1],pics[0].shape[2]))
    for pic in pics:
        img=img+pic
    img=img*1.0/len(pics)
    
    return img


image=[]

for jpg in os.listdir(fpath):
    ima=io.imread(os.path.join(fpath,jpg))
    image.append(ima)

m_pic=meanpic(image)
m_pic=m_pic
image=np.array(image)
image=image
print('m_pic=',m_pic.shape)
print('ima=',image.shape)
image=np.reshape(image,(len(image),-1))
m_pic=np.reshape(m_pic,(1,-1))



minu_i=image-m_pic  

U,s,V=np.linalg.svd(minu_i.transpose(), full_matrices=False)
print(s)
print('U=',U.shape)
print('V=',V.shape)




targetimg=io.imread(tag_path)

recon=np.zeros((600,600,3))
for i in range(4):
    '''
    Uout=U[:,i]-np.min(U[:,i])
    Uout/=np.max(Uout)
    Uout=(Uout*255).astype(np.uint8)
    io.imsave('eig%d.png'%(i),Uout.reshape(600,600,3))
    '''
    
    lamd=np.dot(targetimg.reshape(-1),U[:,i])
    print('lamd=',lamd)
    recon+=lamd*U[:,i].reshape(600,600,3)
    
recon=recon#+m_pic.reshape(600,600,3)  
recon=recon-np.min(recon)
recon/=np.max(recon)
recon=(recon*255).astype(np.uint8)        
io.imsave('reconstruction.jpg',recon/np.max(recon))    
