import numpy as np
#test=np.loadtxt('test.csv',delimiter =',',skiprows=1)
test=np.load('test.npy')
train=np.load('train.npy')
#randomize

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
    r=Embedding(input_dim=int(np.amax(train[:,1])+1),input_length=1,output_dim=40,name='user_em')(u)
    p=Embedding(input_dim=int(np.amax(train[:,2])+1),input_length=1,output_dim=40,name='movie_em')(m)

    '''
    cat=Concatenate()([r,p])
    d=Dense(units=100,use_bias=True,activation='relu')(cat)
    output=Dense(units=1,use_bias=True,activation='sigmoid')(d)
    '''
    
    r=Flatten()(r)
    p=Flatten()(p)
    
    
    dot=Dot(axes=1,name='dot')([r,p])
    #dot=Flatten()(dot)
    rb=Embedding(input_dim=int(np.amax(train[:,1])+1),input_length=1,output_dim=3,name='user_bias')(u)
    pb=Embedding(input_dim=int(np.amax(train[:,2])+1),input_length=1,output_dim=3,name='movie_bias')(m)
    rb=Flatten()(rb)
    pb=Flatten()(pb)
    
    rb=Dense(units=1,activation='relu')(rb)
    pb=Dense(units=1,activation='relu')(pb)
    add=Add()([pb,dot,rb])   
  #  o=Flatten()(add)
    out=Dense(units=1,activation='relu')(add)
   
 #   cat=Concatenate()([rb,pb,dot])
    #out=Dense(units=1,activation='relu')(pb)
    model=Model(inputs=[u,m],outputs=[add])

    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    print(dot.get_shape())
    print(rb.get_shape())
    print(pb.get_shape())
    model.fit([user,movie],rate, validation_data=([userv,moviev], ratev),
             epochs=7, batch_size=32,callbacks=[ModelCheckpoint('model',monitor='val_acc',save_best_only=True)])

         
         
#    model.save('model')
else:
    model=load_model('model')
model=load_model('model')
out=model.predict([test[:,1],test[:,2]])

#np.clip(a, 1, 8)
#outy=out
outy=np.clip(out,0,5)

for i in range(len(outy)):
    if outy[i]>5:
        print('i=',outy[i])
with open('predict.csv','w') as o:
    o.write('TestDataID,Rating\n')
    for i in  range(len(outy)):
        o.write('%d,%f\n' %(i+1,outy[i] ))

