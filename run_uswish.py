from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle
execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET    = sys.argv[-1]
learn_beta = int(sys.argv[-2])
bn         = int(sys.argv[-3])
lr      = 0.001#float(sys.argv[-3])





for model,model_name in zip([DenseCNN,largeCNN],['SmallCNN','LargeCNN']):
	data = dict()
	name   = DATASET+'_'+model_name+'_lb'+str(learn_beta)+'_bn'+str(bn)+'_uswish.pkl'
	for lr in [0.005,0.0001]:
		if('Small' in model_name):
			ne=150
		else:
			ne=150
		if(1):#for bn in [1,0]:#if(1):#for use_beta in [0,1]:
			data[str(lr)]=[[],[],[]]
                        for k in xrange(3):
				x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)
####
				m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',nonlinearity='swish')
				model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
				train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
				temp = model1.get_templates(x_train[:200])
				preds = model1.predict(x_train[:200])
                                data[str(lr)][0].append([train_loss,train_accu,test_accu,temp,preds])
####
                                m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',nonlinearity='uswish')
                                model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
                                train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
                                temp = model1.get_templates(x_train[:200])
                                preds = model1.predict(x_train[:200])
                                data[str(lr)][1].append([train_loss,train_accu,test_accu,temp,preds])
####
                                m = model(bn=bn,n_classes=10,global_beta=1,pool_type='MAX',nonlinearity='smooth')
                                model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=learn_beta)
                                train_loss,train_accu,test_accu = model1.fit(x_train,y_train,x_test,y_test,n_epochs=ne,return_train_accu=1)
                                temp = model1.get_templates(x_train[:200])
                                preds = model1.predict(x_train[:200])
                                data[str(lr)][2].append([train_loss,train_accu,test_accu,temp,preds])
####
        f = open('/mnt/project2/rb42Data/SMASO/'+name,'wb')
        cPickle.dump(data,f)
        f.close()
		
	
	


