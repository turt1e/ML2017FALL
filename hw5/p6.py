import numpy as np

user=np.loadtxt('users.csv',delimiter ='::',skiprows=1,dtype='str')
np.save('user',user)
'''
test=np.loadtxt('test.csv',delimiter =',',skiprows=1)
train=np.loadtxt('train.csv',delimiter =',',skiprows=1)
np.save('test',test)
np.save('train',train)
'''
user=np.load('user.npy')
test=np.load('test.npy')
train=np.load('train.npy')

ud={}
for i in range(len(user)):
    ud.update({int(user[i,0]):user[i,1]})

#randomize
gender=np.zeros((len(train),2))
for i in range(len(train)):
    gender[i,0]=(ud[int(train[i,1])]=='M')*1
    if user[i,1]=='M':
        print(user[i,0],"=='M'")
    gender[i,1]=(ud[int(train[i,1])]=='F')*1

    
train= np.append(train,gender, axis=1)
    
for i in range(10):
    print(train[i])
rand=np.arange(len(train))
for i in range(10):
    print(rand[i])
np.random.shuffle(rand)
for i in range(10):
    print(rand[i])
    
train=np.array(train[rand],dtype='int')


#valid    
val_per=0.9

val_len=  int(val_per*len(train)  )
#normalize
'''
r=train[:,3]
mean=np.mean(r)
sig=np.sqrt(np.mean(np.power((r-mean),2)))
r=(r-mean)/sig
train[:,3]=r
'''
#trainset
traint=train[:val_len]   
user=traint[:,1]
movie=traint[:,2]
rate=traint[:,3]
#rate=np.log((rate+1)/0.1)


for i in range(10):
    print (user[i],',',movie[i],',',rate[i])
    
#validset
trainv=train[val_len:] 
userv=trainv[:,1]
moviev=trainv[:,2]
ratev=trainv[:,3]
#ratev=np.log((ratev+1)/0.1)
print(np.amax(train[:,1]))
print(np.amax(train[:,2]))
#print(sig)
#==============model=============


from keras.models import load_model
from keras.regularizers import l2
from keras.layers import Dense
from keras.layers import Flatten, merge
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.engine import Input

from keras.layers import Dense, Input, Flatten,Dropout,Dot,Concatenate,Add
from keras.models import Model

go=1
if go==1:

   #=><><><><><><>checkpoint><><><><><
    earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath='model.h5', 
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_loss',
                                     mode='min' )
    #==^^^^^^^checkpoint^^^^^^^^


    u=Input(shape=(1,),name='user')
    m=Input(shape=(1,),name='movie')
    man=Input(shape=(1,),name='man')
    female=Input(shape=(1,),name='female')
    r=Embedding(input_dim=int(np.amax(train[:,1])+1),input_length=1,output_dim=40,name='user_em')(u)
    p=Embedding(input_dim=int(np.amax(train[:,2])+1),input_length=1,output_dim=40,name='movie_em')(m)

    '''
    cat=Concatenate()([r,p])
    d=Dense(units=100,use_bias=True,activation='relu')(cat)
    output=Dense(units=1,use_bias=True,activation='sigmoid')(d)
    '''
    
    r=Flatten()(r)
    p=Flatten()(p)
    con=Concatenate()([r,p,man,female])
    dot=Dense(units=1,activation='relu')(con)
    r=Dense(units=40,activation='relu')(r)
    p=Dense(units=40,activation='relu')(p)
    cat=Concatenate()([r,p])
    bias=Dense(units=1,activation='relu')(cat)


    add=Add()([bias,dot])   
  #  o=Flatten()(add)
    out=Dense(units=1,activation='relu')(add)
   
 #   cat=Concatenate()([rb,pb,dot])
    #out=Dense(units=1,activation='relu')(pb)
    model=Model(inputs=[u,m],outputs=[add])

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    print(dot.get_shape())
 #   print(rb.get_shape())
  #  print(pb.get_shape())
    model.fit([user,movie],rate, validation_data=([userv,moviev], ratev),
             epochs=20, batch_size=1000,callbacks=[ModelCheckpoint('model',monitor='val_acc',save_best_only=True)])
    model.summary()
         
         
#    model.save('model')
else:
    model=load_model('model')
model=load_model('model')

gendert=np.zeros((len(test),2))
for i in range(len(test)):
    gendert[i,0]=(ud[int(test[i,1])]=='M')*1
    if (user[i,1]=='M')&&(i<10):
        print(user[i,0],"=='M'")
    gendert[i,1]=(ud[int(test[i,1])]=='F')*1

    

out=model.predict([test[:,1],test[:,2],gendert[:,0],gendert[:,1]])

#np.clip(a, 1, 8)
#outy=out*sig+mean
outy=np.clip(out,0,5)

for i in range(len(outy)):
    if outy[i]>5:
        print('i=',outy[i])
with open('predict.csv','w') as o:
    o.write('TestDataID,Rating\n')
    for i in  range(len(outy)):
        o.write('%d,%f\n' %(i+1,outy[i] ))