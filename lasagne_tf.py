import tensorflow as tf

############################################################################################
#
# OPTIMIZER and LOSSES
#
#
############################################################################################


def resnetnonlinearity(x):
	return tf.maximum(x,2*x)


def ortho_loss2(W):
    return tf.reduce_mean(tf.pow(tf.matmul(W,W,transpose_a=True)-tf.matrix_diag(tf.reduce_mean(W*W,axis=0)),2))

def ortho_loss4(W):
    return tf.reduce_mean(tf.pow(tf.tensordot(W,W,[[0,1,2],[0,1,2]])-tf.matrix_diag(tf.reduce_mean(W*W,axis=[0,1,2])),2))

def categorical_crossentropy(logits, labels):
	labels = tf.cast(labels, tf.int32)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')
	return cross_entropy


def l1_penaly():
	Ws = tf.get_collection("regularizable")
	cost = tf.add_n([tf.norm(v,ord=1) for v in Ws])/float32(len(Ws))
	return cost


def l2_penaly():
        Ws = tf.get_collection("regularizable")
        cost = tf.add_n([tf.norm(v,ord=2) for v in Ws])/float32(len(Ws))
        return cost



def count_number_of_params():
	print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


###########################################################################################
#
#
#		Layers
#
#
###########################################################################################





class Pool2DLayer:
        def __init__(self,incoming,window,pool_type='MAX',global_beta=1):
                self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/window,incoming.output_shape[2]/window,incoming.output_shape[3])
		if(global_beta):
                	self.beta = tf.Variable(tf.zeros(1),trainable=False,name='beta')
		else:
                        self.beta = tf.Variable(tf.zero(incoming.output_shape[-1]),trainable=False,name='beta')
		if(pool_type=='BETA'):
	                tf.add_to_collection('beta',self.beta)
	                beta  = tf.clip_by_value(tf.sigmoid(self.beta),0.1,0.9)
			coeff = beta/(1-beta)
			if(global_beta):
				if(incoming.output_shape>2):
					coeff = tf.reshape(coeff,(1,1,1,1))
				else:
                                        coeff = tf.reshape(coeff,(1,1))
			else:
                                if(incoming.output_shape>2):
                                        coeff = tf.reshape(coeff,(1,1,1,incoming.output_shape[-1]))
                                else:
                                        coeff = tf.reshape(coeff,(1,incoming.output_shape[-1]))
		if(pool_type=='MAX' or pool_type=='AVG'):
			self.output = tf.nn.pool(incoming.output,(window,window),pool_type,padding='VALID',strides=(window,window))
		else:
			beta  = tf.sigmoid(self.beta)
			self.output = tf.nn.pool(incoming.output,(window,window),strides=(window,window),pooling_type='MAX',padding='VALID')*beta+(1-beta)*tf.nn.pool(incoming.output,(window,window),strides=(window,window),pooling_type='AVG',padding='VALID')




class InputLayer:
	def __init__(self,input_shape,x):
		self.output = x
		self.output_shape = input_shape

class DenseLayer:
	def __init__(self,incoming,n_output,training,bn=0,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
		print incoming.output_shape
		if(len(incoming.output_shape)>2):
			inputf = tf.layers.flatten(incoming.output)
			in_dim = prod(incoming.output_shape[1:])
		else:
			inputf = incoming.output
			in_dim = incoming.output_shape[1]
                self.W = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
                tf.add_to_collection("regularizable",self.W)
		self.output_shape = (incoming.output_shape[0],n_output)
		if(init_b=='fixed'):
			self.b = -tf.reduce_sum(self.W*self.W,axis=0,keep_dims=True)*0.5
		else:
                       	self.b = tf.Variable(init_b((1,n_output)),name='b_dense',trainable=True)
		self.output = tf.matmul(inputf,self.W)+self.b



class VQDenseLayer:
	def __init__(self,incoming,n_output,training,R=2,bn=0,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
		print incoming.output_shape
		if(len(incoming.output_shape)>2):
			inputf = tf.layers.flatten(incoming.output)
			in_dim = prod(incoming.output_shape[1:])
		else:
			inputf = incoming.output
			in_dim = incoming.output_shape[1]
                self.W  = tf.Variable(init_W((in_dim,n_output,R)),name='W_dense',trainable=True)
                self.WV = tf.Variable(init_W((in_dim,n_output,R)),name='WV_dense',trainable=True)
#                tf.add_to_collection("regularizable",self.W)
		self.output_shape = (incoming.output_shape[0],n_output)
		inputf = tf.layers.batch_normalization(inputf,training=training,fused=True,trainable=False) 
		if(1):
                        self.bv = -tf.reduce_sum(self.WV*self.WV,axis=0,keep_dims=True)*0.5
                        self.b = tf.Variable(init_W((1,n_output,R)),name='b_dense',trainable=True)
			self.preoutputv = tf.tensordot(inputf,self.WV,[[1],[0]])+self.bv
                        self.preoutput  = tf.tensordot(inputf,self.W,[[1],[0]])+self.b-tf.expand_dims(tf.reduce_sum(self.W*self.WV,0),0)
                self.VQ_shape = (incoming.output_shape[0],n_output,R)
                self.VQ       = tf.nn.softmax(self.preoutputv,dim=2)
		self.output   = tf.reduce_sum(self.VQ*self.preoutput,axis=2)


class VQConv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,R=2,stride=1,pad='valid',mode='CONSTANT',bn=0,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.)):
		print incoming.output_shape
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
		padded_input = tf.pad(tf.expand_dims(padded_input,1),[[0,0],[R-1,R-1],[0,0],[0,0],[0,0]])
                self.W     = tf.Variable(init_W((R,filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
                self.WV    = tf.Variable(init_W((R,filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
                padded_input = tf.layers.batch_normalization(padded_input,training=test,fused=True)
                output1    = tf.nn.conv3d(padded_input,self.WV,strides=[1,1,stride,stride,1],padding='VALID')
#                output2    = tf.nn.conv3d(padded_input,self.W,strides=[1,1,stride,stride,1],padding='VALID')
                if(1):
			self.b      = tf.Variable(init_b((1,R,1,1,n_filters)),name='b_conv',trainable=True)
                        self.bv     = -tf.reshape(tf.reduce_sum(self.WV*self.WV,axis=[1,2,3]),(1,R,1,1,n_filters))*0.5
                        self.preoutput1 = output1+self.bv
                        self.preoutput2 = self.b#output2+self.b#-tf.reshape(tf.reduce_sum(self.W*self.WV,axis=[1,2,3]),(1,R,1,1,n_filters))
                tf.add_to_collection("regularizable",self.W)
                self.VQ_shape = (incoming.output_shape[0],R)+self.output_shape[1:]
                self.VQ       = tf.nn.softmax(self.preoutput1,dim=1)
		self.output   = tf.reduce_sum(self.VQ*self.preoutput2,axis=1)







class NonlinearityLayer:
        def __init__(self,incoming,nonlinearity,training,use_beta=0,global_beta=1,bn=1):
                self.output_shape = incoming.output_shape
		if(bn):
			output = tf.layers.batch_normalization(incoming.output,training=training,fused=True)
		else:
			output = incoming.output
		if(nonlinearity=='relu'):
			self.output = tf.nn.relu(output)
		elif(nonlinearity=='lrelu'):
			self.output = tf.nn.lrelu(output)
		elif(nonlinearity=='abs'):
			self.output = tf.abs(output)
		elif(nonlinearity=='swish'):
			self.eta = tf.Variable(tf.ones(1)*0.5,trainable=False,name='beta')
			tf.add_to_collection('beta',self.eta)
			self.output= tf.nn.sigmoid(output*tf.nn.softplus(self.eta))*output
		elif(nonlinearity=='uswish'):
			self.eta = tf.Variable(tf.ones(1)*0.5,trainable=False,name='beta')
			tf.add_to_collection('beta',self.eta)
			self.output = tf.nn.sigmoid(tf.stop_gradient(output)*self.eta)*output
		elif(nonlinearity=='smooth'):
                        self.eta = tf.Variable(tf.ones(1)*0.5,trainable=False,name='beta')
                        tf.add_to_collection('beta',self.eta)
                        self.output = tf.log(1+tf.exp(output*self.eta))/(self.eta+0.0000000001)
#		if(global_beta):
#                	self.beta = tf.Variable(tf.zeros(1),trainable=False,name='beta')
#		else:
#                        self.beta = tf.Variable(tf.zeros(incoming.output_shape[-1]),trainable=False,name='beta')
#		if(use_beta):
#                	tf.add_to_collection('beta',self.beta)
#		beta  = tf.sigmoid(self.beta)
#		if(use_beta):
#			coeff = beta/(1-beta)
#			if(global_beta):
#				if(len(incoming.output_shape)>2):
#					coeff = tf.reshape(coeff,(1,1,1,1))
#				else:
#                                        coeff = tf.reshape(coeff,(1,1))
#			else:
#                                if(len(incoming.output_shape)>2):
#                                        coeff = tf.reshape(coeff,(1,1,1,incoming.output_shape[-1]))
#                                else:
#                                        coeff = tf.reshape(coeff,(1,incoming.output_shape[-1]))
#			if(nonlinearity=='relu'):
#	        	        self.output = tf.nn.sigmoid(incoming.output)*incoming.output
#			elif(nonlinearity=='lrelu'):
#				coeff1 = tf.exp(tf.clip_by_value(coeff*0.01*incoming.output,-10,10))
#				coeff2 = tf.exp(tf.clip_by_value(coeff*incoming.output,-10,10))
#				mask1  = coeff1/(coeff1+coeff2)
#				mask2  = coeff2/(coeff1+coeff2)
#                	        self.output = incoming.output*(mask1*0.01+mask2)
#			elif(nonlinearity=='abs'):
#				coeff1 = tf.exp(-coeff*incoming.output)
#				coeff2 = tf.exp(coeff*incoming.output)
#                	        self.output = incoming.output*(-coeff1+coeff2)/(coeff1+coeff2)
#		else:
#                        if(nonlinearity=='relu'):
#                                self.output = tf.nn.relu(incoming.output)
#                        elif(nonlinearity=='lrelu'):
#                                self.output = tf.nn.leaky_relu(incoming.output)
#                        elif(nonlinearity=='abs'):
#                                self.output = tf.nn.softplus(incoming.output)


#                self.VQ  = tf.stack([mask1,mask2],axis=2)
#		self.VQ_shape = tuple(self.output_shape[:2])+tuple([2])+tuple(self.output_shape[2:])
			



class GlobalPoolLayer:
        def __init__(self,incoming,pool_type='AVG',global_beta=1):
                self.output = tf.reduce_mean(incoming.output,[1,2],keep_dims=True)
                self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]
		self.pool_layer = Pool2DLayer(incoming,incoming.output_shape[1],pool_type,global_beta=global_beta)


class VConv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',bn=0,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),first=False):
		print incoming.output_shape
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                self.W     = tf.Variable(init_W((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
		if(0):
#		padded_input = tf.layers.batch_normalization(padded_input,training=test,fused=True)
	                output1    = tf.nn.conv2d(padded_input,tf.nn.softplus(self.W),strides=[1,stride,stride,1],padding='VALID')
	                if(1):
				self.b      = tf.Variable(-tf.reshape(tf.reduce_sum(tf.nn.softplus(self.W)*tf.nn.softplus(self.W),axis=[0,1,2]),(1,1,1,n_filters))*0.5)#tf.Variable(init_b((1,1,1,n_filters)),name='b_conv',trainable=True)
	                        output = output1+self.b
				self.VQ     = tf.nn.sigmoid(output)
				self.output = self.VQ*output
	                else:
	                        self.output = tf.layers.batch_normalization(output1,training=test,fused=True)
	                tf.add_to_collection("regularizable",self.W)
			print self.output_shape
		else:
                        output1    = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
                        if(1):
                                self.b      = tf.Variable(-tf.reshape(tf.reduce_sum(self.W*self.W,axis=[0,1,2]),(1,1,1,n_filters))*0.5)#tf.Variable(init_b((1,1,1,n_filters)),name='b_conv',trainable=True)
                                output = output1+self.b
                                self.VQ     = tf.nn.sigmoid(output)
                                self.output = self.VQ*output
                        else:
                                self.output = tf.layers.batch_normalization(output1,training=test,fused=True)
                        tf.add_to_collection("regularizable",self.W)
                        print self.output_shape



class Conv2DLayer:
        def __init__(self,incoming,n_filters,filter_shape,test,stride=1,pad='valid',mode='CONSTANT',bn=0,init_W = tf.contrib.layers.xavier_initializer(uniform=True),init_b = tf.constant_initializer(0.),first=False):
		print incoming.output_shape
                if(pad=='valid' or filter_shape==1):
                        padded_input = incoming.output
                        self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)/stride,(incoming.output_shape[1]-filter_shape+1)/stride,n_filters)
                elif(pad=='same'):
                        assert(filter_shape%2 ==1)
                        p = (filter_shape-1)/2
                        padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]/stride,incoming.output_shape[2]/stride,n_filters)
                else:
                        p = filter_shape-1
                	padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
                	self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)/stride,(incoming.output_shape[1]+filter_shape-1)/stride,n_filters)
                self.W      = tf.Variable(init_W((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
                output1     = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
                self.b      = tf.Variable(tf.zeros((1,1,1,n_filters)),trainable=True)#tf.Variable(init_b((1,1,1,n_filters)),name='b_conv',trainable=True)
                self.output = output1+self.b
                tf.add_to_collection("regularizable",self.W)
                print self.output_shape










