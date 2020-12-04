#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import time

# Size here doesn't matter, it's just needed to derive the object length
a_np = np.random.rand(1024).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
__kernel void empty()
{

}
""").build()

# Warmup
prg.empty(queue, a_np.shape, None)
prg.empty(queue, a_np.shape, None)
prg.empty(queue, a_np.shape, None)
prg.empty(queue, a_np.shape, None)
queue.finish()


# Start the actual timing
start = time.time()

for i in range(1000):
    prg.empty(queue, a_np.shape, None)
    
end = time.time()

print((end - start))

