from pylab import *
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import cPickle


execfile('utils.py')
execfile('models.py')
execfile('lasagne_tf.py')


import matplotlib as mpl
#mpl.rc('text', usetex=True)
#mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
#mpl.rcParams.update(pgf_with_rc_fonts)
label_size = 25
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

from scipy.signal import convolve2d
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.datasets import make_moons,make_circles


#DATASET = sys.argv[-1]
lr      = 0.0001#float(sys.argv[-3])





#x_train,x_test,y_train,y_test,c,n_epochs,input_shape=load_utility(DATASET)



X,y   = make_moons(10000,noise=0.035,random_state=20)
x_,y_ = make_circles(10000,noise=0.02,random_state=20)
x_[:,1]+= 2.
y_   += 2
X     = concatenate([X,x_],axis=0)
y     = concatenate([y,y_])
X    -= X.mean(0,keepdims=True)
X    /= X.max(0,keepdims=True)

print y
X=X.astype('float32')
y=y.astype('int32')
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,stratify=y,random_state=20)
#Ns=[int(sys.argv[-2]),int(sys.argv[-1]),10]
input_shape = (100,2)
print shape(x_train),shape(y_train),shape(x_test),shape(y_test)



h=500
x_min, x_max = X[:, 0].min() - .15, X[:, 0].max() + .15
y_min, y_max = X[:, 1].min() - .15, X[:, 1].max() + .15
xx, yy = np.meshgrid(linspace(x_min, x_max, h),linspace(y_min, y_max, h))
xxx=xx.flatten()
yyy=yy.flatten()
DD = asarray([xxx,yyy]).astype('float32').T


for D1 in [45]:
	for R in [3]:
		for bn in [1]:
#                        name = DATASET+'_'+model_name+'_bn'+str(bn)+'_'+nonlinearity+'_VQmoons.pkl'
			m = VQlinear(bn=bn,n_classes=2,D1=D1,R=3,global_beta=1,pool_type='MAX',use_beta=1)
			model1    = VQClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=0)
			updates   = set_betas(float32(0))
			model1.session.run(updates)
			train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=8)
#			VQ = model1.getVQ(x_train)
#			preds = model1.predict(x_train)
#			values0 = VQ2values(VQ[0])
#
			m = linear(bn=bn,n_classes=2,D1=D1,R=3,global_beta=1,pool_type='MAX',use_beta=1)
			model1    = DNNClassifier(input_shape,m,optimizer = tf.train.AdamOptimizer,lr=lr,learn_beta=0)
			updates   = set_betas(float32(0))
			model1.session.run(updates)
			train_loss_pre,test_loss_pre = model1.fit(x_train,y_train,x_test,y_test,n_epochs=8)
#			VQ = model1.getVQ(x_train)
#			preds = model1.predict(x_train)
#                        values1 = VQ2values(VQ[0])
#	
	
	


