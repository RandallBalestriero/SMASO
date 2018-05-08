from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle

execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET    = sys.argv[-1]
#learn_beta = int(sys.argv[-2])
bn         = int(sys.argv[-2])
#lr         = 0.001#float(sys.argv[-3])
learn_beta=1




for model,model_name in zip([DenseCNN,largeCNN],['SmallCNN','LargeCNN']):
	data = dict()
	for lr in [0.001,0.0002]:
	        name   = DATASET+'_'+model_name+'_lr'+str(lr)+'_bn'+str(bn)+'_ortho'
		if('Small' in model_name):
			ne=75
		else:
			ne=75
		if(1):#for bn in [1,0]:#if(1):#for use_beta in [0,1]:
                        for k in xrange(5):
				x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
####
				m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',nonlinearity='swish',centered=0,ortho=0)
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
				train_loss,train_accu,test_accu,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
				temp = model1.get_templates(x_train[:500])
				preds = model1.predict(x_train[:500])
                                data0=[train_loss,train_accu,test_accu,temp,preds,W]
####
                                m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',nonlinearity='swish',centered=1,ortho=1)
                                model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
                                train_loss,train_accu,test_accu,W = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
                                temp = model1.get_templates(x_train[:500])
                                preds = model1.predict(x_train[:500])
                                data1=[train_loss,train_accu,test_accu,temp,preds,W]
####
			        f = open('/mnt/project2/rb42Data/SMASO/'+name+str(k)+'.pkl','wb')
			        cPickle.dump([data0,data1],f)
			        f.close()
		
	
	


