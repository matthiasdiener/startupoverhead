#!/usr/bin/env python
from time import time

import numpy as np
import pyopencl as cl

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
#queue = cl.CommandQueue(ctx, properties=cl.QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
nruns = 50000


def doit():
    knl = prg.sum
    start = time()
    for i in range(nruns):
        knl(queue, (1,), None, a_g, b_g, res_g)
    elapsed = time()-start
    print(f"{elapsed/nruns:.3e}s per enqueue, {elapsed}s total")


if 0:
    from pyinstrument import Profiler
    profiler = Profiler()
    profiler.start()
    doit()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
else:
    doit()
