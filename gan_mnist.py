"""

Code referenced from https://github.com/AYLIEN/gan-intro and https://github.com/bamos/dcgan-completion.tensorflow

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.pyplot import cm



# linear operation of multiplication of input vector with the weight vector
def linear(input_,output_dim,scope = None,stddev = 0.02, bias_start = 0.0, with_w = False):
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(scope or 'linear'):
        matrix = tf.get_variable("matrix",[shape[1],output_dim],tf.float32,tf.random_normal_initializer(stddev = stddev))
        bias = tf.get_variable("bias",[output_dim],initializer = tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_,matrix)+bias,matrix,bias
        else:
            return tf.matmul(input_,matrix)+bias

# 2d convolution on the input image with kernel size 3x3 and strides of 2x2
def conv2d(input_,output_dim,k_w = 3,k_h = 3,s_w = 2,s_h = 2,stddev = 0.02,name = "conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[k_h,k_w,input_.get_shape()[-1],output_dim],initializer = tf.truncated_normal_initializer(stddev = stddev))
        conv = tf.nn.conv2d(input_,w,strides = [1,s_h,s_w,1],padding = 'SAME')
        biases = tf.get_variable('bias',[output_dim],initializer = tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv,biases)
        return conv

# implementation of leaky recified linear output unit
def lrelu(x,leak = 0.2,name = 'lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1 * x + f2 * abs(x)
        
# function to implement convolution transpose for the generator
def conv2d_transpose(input_,output_dim,k_w = 3,k_h = 3,s_w = 2,s_h = 2,stddev = 0.02,name = "conv2d_transpose",with_w = False):
    with tf.variable_scope(name):
        w = tf.get_variable('w',[k_h,k_w,output_dim[-1],input_.get_shape()[-1]],initializer = tf.random_normal_initializer(stddev = stddev))
        deconv = tf.nn.conv2d_transpose(input_,w,output_shape = output_dim,strides = [1,s_h,s_w,1])
        biases = tf.get_variable('bias',[output_dim[-1]],initializer = tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv,biases)
        if with_w:
            return deconv,w,biases
        else:
            return deconv

# defining the discriminator of the Generative Adversarial Network
def discriminator(image,d_dim=32,reuse = False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    
    h0 = lrelu(conv2d(image,d_dim,name = 'd_h0_conv'))
    h1 = lrelu(conv2d(h0,d_dim*2,name = 'd_h1_conv'))
    h2 = lrelu(conv2d(h1,d_dim*4,name = 'd_h2_conv'))
    h3 = linear(tf.reshape(h2,[-1,1024]),1,'d_h3_linear')
    
    return tf.nn.sigmoid(h3),h3

# defining the generator of the Generative Adversarial Network
def generator(z,gf_dim = 32,batch_size = 32):
    z_,h0_w,h0_b = linear(z,gf_dim*4*4*4,'g_h0_lin',with_w = True)
    
    h0 = tf.reshape(z_,[-1,4,4,gf_dim*4])
    h0 = tf.nn.relu(h0)
    
    h1,h1_w,h1_b = conv2d_transpose(h0,[batch_size,7,7,gf_dim*2],name = 'g_h1',with_w = True)
    h1 = tf.nn.relu(h1)
    
    h2,h2_w,h2_b = conv2d_transpose(h1,[batch_size,14,14,gf_dim*1],name = 'g_h2', with_w = True)
    h2 = tf.nn.relu(h2)
    
    h3, h3_w,h3_b = conv2d_transpose(h2,[batch_size,28,28,1],name = 'g_h3',with_w = True)
    return tf.nn.tanh(h3)
    
# putting the parts together in a single class
class GAN(object):
    def __init__(self,data,z_dim = 100,gf_dim = 32,d_dim = 32,image_size = 28,num_steps = 100,batch_size = 32,learning_rate = 0.0002,beta_for_adam = 0.5):
        self.data = data
        self.image_size = image_size
        self.image_shape = [self.image_size,self.image_size,1]
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.d_dim = d_dim
        self.learning_rate = learning_rate
        self.beta_for_adam = beta_for_adam
        self.create_model()
        
        
    # creating placeholders for generator and discriminator
    def create_model(self):
        self.images = tf.placeholder(tf.float32,[None]+self.image_shape,name = 'real_images')
        self.z = tf.placeholder(tf.float32,[None,self.z_dim],name = 'z')
        
        self.G = generator(self.z,self.gf_dim,self.batch_size)
        self.D,self.D_logits = discriminator(self.images,self.d_dim)
        
        self.D_,self.D_logits_ = discriminator(self.G,self.d_dim,reuse = True)
        
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,tf.zeros_like(self.D_)))
        
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,tf.ones_like(self.D_)))
        
        self.d_loss = self.d_loss_fake + self.d_loss_real
        
        t_vars = tf.trainable_variables()
        
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(self.learning_rate,beta1 = self.beta_for_adam).minimize(self.d_loss,var_list = self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.learning_rate,beta1 = self.beta_for_adam).minimize(self.g_loss,var_list = self.g_vars)
        
    # training the model using adam optimizer
    def train(self):
        with tf.Session() as session:
            session.run(tf.initialize_all_variables())
            step = 0
            count = 0
            while(count<self.num_steps):
                if((step+1)*self.batch_size > 54999):
                    step = 0
                batch_files = self.data[step*self.batch_size:(step+1)*self.batch_size,:]
                batch = np.reshape(batch_files,[-1,28,28,1])
                batch_images = np.array(batch).astype(np.float32)
                
                batch_z = np.random.uniform(-1,1,[self.batch_size,self.z_dim]).astype(np.float32)
                
                loss_d_real,loss_d_fake,_ = session.run([self.d_loss_real,self.d_loss_fake,self.d_optim],\
                                                        feed_dict = {self.images:batch_images,self.z:batch_z})
                
                loss_g = session.run([self.g_loss,self.g_optim],
                                     feed_dict = {self.z:batch_z})
                
                loss_g = session.run([self.g_loss,self.g_optim],
                                     feed_dict = {self.z:batch_z})
                
                if(count%500 == 0):
                    print(count," ",loss_d_real+loss_d_fake," ",loss_g)
                    # generating images after training the model 
                    sample = np.random.uniform(-1,1,[self.batch_size,self.z_dim]).astype(np.float32)
                    g = session.run(self.G,feed_dict = {self.z: sample})
                    g = np.reshape(g,[self.batch_size,784])
                    for i in range(self.batch_size):
                        img = g[i:i+1,]
                        img = np.reshape(img,[28,28])
                        file_name = "img_after_{}".format(count)+str(i)
                        plt.imsave(file_name,img,cmap = cm.gray)
                
                step = step+1
                count = count+1
                
# extracting the mnist dataset for training
def main():
    learning_rate = 0.0002
    beta_for_adam = 0.5
    z_dim = 300
    gf_dim = 32
    d_dim = 32
    image_size = 28
    num_steps = 10001
    batch_size = 32
    mnist = input_data.read_data_sets("MNIST",one_hot = True)
    x_train = mnist.train.images
    model = GAN(x_train,z_dim,gf_dim,d_dim,image_size,num_steps,batch_size,learning_rate,beta_for_adam)
    model.train()
    
if __name__ == '__main__':
    main()

