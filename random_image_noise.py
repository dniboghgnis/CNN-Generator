import tensorflow as tf
import numpy as np

tf.reset_default_graph()
   # To clear the defined variables and operations of the previous cell
# create graph
import matplotlib.pyplot as plt

#create white image with noise for generator
#z_image=cv2.imread('/Users/Gobind/TransAtlantic/1TF_dataScience/Datasets/noisy_white_image.jpg')
z_image=[]
for i in range(7500):
    temp=np.random.randint(255)
    z_image.append(temp)
#print z_image
# create graph

z_image=tf.reshape(z_image,[50,50,3])

#print(z_image)
#z_image=np.array(z_image,dtype=np.int64)
#noisy_white_array=[]
##z_image=np.array(z_image,dtype=np.float64)
##z_image=cv2.resize(z_image,(50,50))
#noisy_white_array.append(z_image)
##noisy_white_array=noisy_white_array%255
#print("Noisy image shape       "+str(z_image.shape))
##plt.imshow(noisy_white_array[0])
##plt.show()
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
#    print(sess.run(c))
    photu=sess.run(z_image)
#    print(photu)
    plt.imshow(photu)
    plt.show()

    