from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle
execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')

DATASET = sys.argv[-1]
lr      = 0.0005#float(sys.argv[-3])





x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)

x_train1,X2,y_train1,y2 = train_test_split(x_train,y_train,train_size=0.8,stratify=y_train)



for model,model_name in zip([DenseCNN],['denseCNN']):
	for nonlinearity in ['relu']:
		for bn in [0,1]:
                        name = DATASET+'_'+model_name+'_bn'+str(bn)+'_'+nonlinearity+'_dual.pkl'
			ALL_TRAIN0 = []
			ALL_TEST0 = []
                        ALL_TRAIN1 = []
                        ALL_TEST1 = []
                        for k in xrange(10):
				m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=1,global_beta=1,pool_type='BETA')
				model1    = DNNClassifierDual(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,dual=1)
				updates   = set_betas(float32(0))
				model1.session.run(updates)
				train_loss_pre,test_loss_pre = model1.fit(x_train1,y_train1,X2,y2,x_test,y_test,n_epochs=80)
				ALL_TRAIN0.append(train_loss_pre)
				ALL_TEST0.append(test_loss_pre)
                                m = model(bn=bn,n_classes=10,nonlinearity=nonlinearity,use_beta=1,global_beta=1,pool_type='BETA')
                                model1    = DNNClassifierDual(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,dual=0)
                                updates   = set_betas(float32(0))
                                model1.session.run(updates)
                                train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,X2,y2,x_test,y_test,n_epochs=80)
                                ALL_TRAIN1.append(train_loss_pre)
                                ALL_TEST1.append(test_loss_pre)
		        f = open('/mnt/project2/rb42Data/SMASO/'+name,'wb')
		        cPickle.dump([asarray(ALL_TRAIN0),asarray(ALL_TEST0),asarray(ALL_TRAIN1),asarray(ALL_TEST1)],f)
		        f.close()
		
	
	


