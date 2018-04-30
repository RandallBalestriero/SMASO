from pylab import *
import matplotlib as mpl
label_size = 15
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size




def smoothrelu(x,beta):
        coeff = beta/(1-beta)
	return log(1+exp(x*coeff))/coeff

def smoothlrelu(x,beta):
        coeff = beta/(1-beta)
        return log(exp(-0.01*coeff*x)+exp(x*coeff))/coeff

def smoothabs(x,beta):
        coeff = beta/(1-beta)
        return log(exp(-coeff*x)+exp(x*coeff))/coeff


def maso(W,b,x):
	return (W.reshape((1,-1))*x.reshape((-1,1))+b.reshape((1,-1))).max(1)

def sigmoid(x):
	return 1/(1+exp(-x))

def dsigmoid(x):
        return sigmoid(x)*(1-sigmoid(x))#1/(1+exp(-x))


def swish(x,beta):
        coeff = beta/(1-beta)
	return sigmoid(coeff*x)*x

def abs(x,beta):
	coeff = beta/(1-beta)
	return (-exp(-coeff*x)+exp(coeff*x))/(exp(-coeff*x)+exp(coeff*x))*x


def lrelu(x,beta):
        coeff = beta/(1-beta)
        return (0.01*exp(0.01*coeff*x)+exp(coeff*x))/(exp(0.01*coeff*x)+exp(coeff*x))*x


def dswish(x,beta):
        coeff = beta/(1-beta)
	return sigmoid(coeff*x)

def dabs(x,beta):
	coeff = beta/(1-beta)
	return (-exp(-coeff*x)+exp(coeff*x))/(exp(-coeff*x)+exp(coeff*x))


def dlrelu(x,beta):
        coeff = beta/(1-beta)
        return (0.01*exp(0.01*coeff*x)+exp(coeff*x))/(exp(0.01*coeff*x)+exp(coeff*x))

def ddswish(x,beta):
        coeff = beta/(1-beta)
        return dsigmoid(coeff*x)*x+sigmoid(coeff*x)


def gg(t,coeff,alpha):
	p1 = (alpha*exp(alpha*coeff*t)+exp(coeff*t))/(exp(alpha*coeff*t)+exp(coeff*t))
	p2 = ((alpha**2*exp(alpha*coeff*t)+exp(coeff*t))*(exp(coeff*alpha*t)+exp(coeff*t))+(alpha*exp(alpha*coeff*t)+exp(coeff*t))**2)/(exp(alpha*coeff*t)+exp(coeff*t))**2
	return t*p2+p1

def ddabs(x,beta):
        coeff = beta/(1-beta)
        return gg(x,coeff,-1)


def ddlrelu(x,beta):
        coeff = beta/(1-beta)
        return gg(x,coeff,0.01)








x=linspace(-4,4,1000)

fs = 2.5

fig=figure(figsize=(18,3.3))

subplot(131)
plot(x,swish(x,0.1),'g',linewidth=fs)
plot(x,swish(x,0.5),'b',linewidth=fs)
plot(x,swish(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
legend([r'$\beta=0.1$',r'$\beta=0.5$',r'$\beta=0.9$'],fontsize=20,loc="upper left")
title('ReLU',fontsize=18)
subplot(132)
plot(x,abs(x,0.1),'g',linewidth=fs)
plot(x,abs(x,0.5),'b',linewidth=fs)
plot(x,abs(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('Abs. Value',fontsize=18)

#legend([r'$\beta=0.1$',r'$\beta=0.5$',r'$\beta=0.9$'],fontsize=20,bbox_to_anchor=(1,1), loc="upper right",bbox_transform=fig.transFigure, ncol=3)

subplot(133)
plot(x,lrelu(x,0.1),'g',linewidth=fs)
plot(x,lrelu(x,0.5),'b',linewidth=fs)
plot(x,lrelu(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('leaky-ReLU',fontsize=18)

tight_layout()

savefig('plot_swish.png')
close()

figure(figsize=(18,3.3))

subplot(131)
plot(x,dswish(x,0.1),'g',linewidth=fs)
plot(x,dswish(x,0.5),'b',linewidth=fs)
plot(x,dswish(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
legend([r'$\beta=0.1$',r'$\beta=0.5$',r'$\beta=0.9$'],fontsize=20,loc="upper left")
title('ReLU',fontsize=18)
subplot(132)
plot(x,dabs(x,0.1),'g',linewidth=fs)
plot(x,dabs(x,0.5),'b',linewidth=fs)
plot(x,dabs(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('Abs. Value',fontsize=18)
subplot(133)
plot(x,dlrelu(x,0.1),'g',linewidth=fs)
plot(x,dlrelu(x,0.5),'b',linewidth=fs)
plot(x,dlrelu(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('leaky-ReLU',fontsize=18)
tight_layout()

savefig('plot_swish_ud.png')
close()


figure(figsize=(18,3.3))

subplot(131)
plot(x,ddswish(x,0.1),'g',linewidth=fs)
plot(x,ddswish(x,0.5),'b',linewidth=fs)
plot(x,ddswish(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
legend([r'$\beta=0.1$',r'$\beta=0.5$',r'$\beta=0.9$'],fontsize=20,loc="upper left")
title('ReLU',fontsize=18)
subplot(132)
plot(x,ddabs(x,0.1),'g',linewidth=fs)
plot(x,ddabs(x,0.5),'b',linewidth=fs)
plot(x,ddabs(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('Abs. Value',fontsize=18)
subplot(133)
plot(x,ddlrelu(x,0.1),'g',linewidth=fs)
plot(x,ddlrelu(x,0.5),'b',linewidth=fs)
plot(x,ddlrelu(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('leaky-ReLU',fontsize=18)
tight_layout()

savefig('plot_swish_bd.png')
close()

figure(figsize=(18,3.3))

subplot(131)
plot(x,smoothrelu(x,0.1),'g',linewidth=fs)
plot(x,smoothrelu(x,0.5),'b',linewidth=fs)
plot(x,smoothrelu(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
legend([r'$\beta=0.1$',r'$\beta=0.5$',r'$\beta=0.9$'],fontsize=20,loc="upper left")
title('ReLU',fontsize=18)
subplot(132)
plot(x,smoothabs(x,0.1),'g',linewidth=fs)
plot(x,smoothabs(x,0.5),'b',linewidth=fs)
plot(x,smoothabs(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on') 
title('Abs. Value',fontsize=18)
subplot(133)
plot(x,smoothlrelu(x,0.1),'g',linewidth=fs)
plot(x,smoothlrelu(x,0.5),'b',linewidth=fs)
plot(x,smoothlrelu(x,0.9),'k',linewidth=fs)
axvline(0,color='k',alpha=0.5)
axhline(0,color='k',alpha=0.5)
grid('on')
title('leaky-ReLU',fontsize=18)
tight_layout()

savefig('plot_smooth.png')



#show()







#show()



