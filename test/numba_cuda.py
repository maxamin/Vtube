import numpy as np,time
import numba
from numba import cuda
import set_env
import math
set_env.set_env()
print("np.__version__",np.__version__)
print("numba.__version__",numba.__version__)
#cuda.detect()

@numba.jit(nopython=True) 
def add_scalars_jit(a, b, c): 
    m=np.linalg.add(a,b)
    return m

def add_scalars(a, b, c): 
    t = a* b
    return t

a=np.random.randn(1001000)
b=np.random.randn(1001000)
c=np.random.randn(1000,1000,3)
add_scalars_jit(a,b,c)


time1=time.time()
add_scalars_jit(a,b,c)

time2=time.time()
print("add_scalars_jit time use:",time2-time1)

add_scalars(a,b,c)
time3=time.time()
print("add_scalars use:",time3-time2)