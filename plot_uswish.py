from pylab import *
import glob
import cPickle

files    = glob.glob('/mnt/project2/rb42Data/SMASO/*ortho*.pkl')
datasets = unique([f.split('_')[0].split('/')[-1] for f in files])

print "files",files
print "datasets",datasets

def formatting(files):
	train_loss0 = []
	test_accu0  = []
	train_accu0 = []
	temp0 = []
	pred0 = []
	W0    = []
        train_loss1 = []
        test_accu1  = []
        train_accu1 = []
        temp1 = []
        pred1 = []
        W1    = []
	for f in files:
		print f
		fi=open(f,'rb')
		b=cPickle.load(fi)
		fi.close()
		train_loss0.append(b[0][0])
		train_accu0.append(b[0][1])
		test_accu0.append(b[0][2])
		temp0.append(b[0][3])
		pred0.append(b[0][4])
		W0.append(b[0][5])
                train_loss1.append(b[1][0])
                train_accu1.append(b[1][1])
                test_accu1.append(b[1][2])
                temp1.append(b[1][3])
                pred1.append(b[1][4])
                W1.append(b[1][5])
	return [stack(train_loss0,0),stack(train_accu0,0),stack(test_accu0,0),concatenate(temp0,0),concatenate(pred0,0),W0[0]],[stack(train_loss1,0),stack(train_accu1,0),stack(test_accu1,0),concatenate(temp1,0),concatenate(pred1,0),W1[0]]


for datas in datasets:
	files  = glob.glob('/mnt/project2/rb42Data/SMASO/'+datas+'*ortho*.pkl')
	models = unique([f.split('_')[1] for f in files])
	print "models",models
	for m in models:
		print "MODEL:",m
	        files_m       = glob.glob('/mnt/project2/rb42Data/SMASO/'+datas+'_'+m+'*ortho*.pkl')
		print files_m
		learning_rate = unique([f.split('_')[2] for f in files_m])
		for lr in learning_rate:
			print "LEARNING RATE",lr
		        files_m_lr  = glob.glob('/mnt/project2/rb42Data/SMASO/'+datas+'_'+m+'_'+lr+'*ortho*.pkl')
			bns    = unique([f.split('_')[3] for f in files_m_lr])
			print files_m_lr
			for bn in bns:
	                        files_m_lr_bn = glob.glob('/mnt/project2/rb42Data/SMASO/'+datas+'_'+m+'_'+lr+'_'+bn+'*ortho*.pkl')
		                data      = formatting(files_m_lr_bn)
				train_loss0,train_accu0,test_accu0,temp0,preds0,W0 = data[0]
                                train_loss1,train_accu1,test_accu1,temp1,preds1,W1 = data[1]
				figure()
				subplot(121)
				plot(train_accu0.mean(0),'b')
	                        plot(test_accu0.mean(0),'--b')
		                plot(train_accu1.mean(0),'k')
		                plot(test_accu1.mean(0),'--k')
#			subplot(122)
#                        plot(train_loss0.mean(0),'b')
#                        plot(train_loss1.mean(0),'k')

#                               tempk =temp- temp.min(axis=(1,2,3,4),keepdims=True)
#                               tempk/= temp.max(axis=(1,2,3,4),keepdims=True)
#                               for k in xrange(9):
#                                       for i in xrange(10):
#                                               subplot(9,10,1+i+k*10)
#                                               imshow(tempk[k][i].mean(2),aspect='auto',cmap='gray',vmin=0,vmax=1)
#                                               title(str(sum(temp[k][i]*x_train[k]))+','+str(preds[k][i]))
#                                               xticks([])
#                                               yticks([])
#                               tight_layout()
#                               show()
				show()

