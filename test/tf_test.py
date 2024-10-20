import os,cv2,logging,time,numpy as np
import set_env
set_env.set_env()
import tensorflow as tf
import tensorflow.keras 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   
print("tf.version:",tf.version.VERSION)
 

def test_mat_calc():
    for i in range(5):
        time.sleep(0.5)
        print(f"------{i}-------------")
        ma=np.random.randn(1000,1000)
        mb=np.random.randn(1000,1000)
        with tf.device("/gpu:0"):
            time1=time.time()
            ta=tf.Variable(ma)
            tb=tf.Variable(mb)
            time2=time.time()
            print("传输gpu变量(ms):",round((time2-time1)*1000))
            
        time1=time.time()
        c=np.matmul(ma,ma)
        d=np.matmul(c,ma)
        e=np.matmul(d,mb)
        time2=time.time()
        print("cpu matmul  time(ms):",round((time2-time1)*1000))

        with tf.device("/gpu:0"):
            time1=time.time()
            #ta=tf.Variable(ma) 
            c=tf.matmul(ta,ta)
            d=tf.matmul(c,ta)
            e=tf.matmul(d,ta)
            time2=time.time()
            print("tf  matmul gpu time(ms):",round((time2-time1)*1000))

            n=e.numpy()
            time3=time.time()
            print("tf  tensor to numpy time(ms):",round((time3-time2)*1000))


test_mat_calc()

 


print("运行结束\n\n")