# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3.1/3.2 Diagnostics Output
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/lia
moreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (167)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (167) 
------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                             | 
        out: Storage,                                                                     | 
        out_shape: Shape,                                                                 | 
        out_strides: Strides,                                                             | 
        in_storage: Storage,                                                              | 
        in_shape: Shape,                                                                  | 
        in_strides: Strides,                                                              | 
    ) -> None:                                                                            | 
        # TODO: Implement for Task 3.1.                                                   | 
        same_shape = len(out_shape) == len(in_shape) and np.all(out_shape == in_shape)----| #0
        same_strides = len(out_strides) == len(in_strides) and np.all(                    | 
            out_strides == in_strides-----------------------------------------------------| #1
        )                                                                                 | 
                                                                                          | 
        # When `out` and `in` are stride-aligned, avoid indexing                          | 
        if same_shape and same_strides:                                                   | 
            for i in prange(len(out)):----------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                | 
        else:                                                                             | 
            for i in prange(len(out)):----------------------------------------------------| #3
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                            | 
                in_index = np.empty(MAX_DIMS, dtype=np.int32)                             | 
                to_index(i, out_shape, out_index)                                         | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                 | 
                                                                                          | 
                in_pos = index_to_position(in_index, in_strides)                          | 
                out_pos = index_to_position(out_index, out_strides)                       | 
                                                                                          | 
                out[out_pos] = fn(in_storage[in_pos])                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/liamoreilly/Desktop
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (187) is hoisted out of
 the parallel loop labelled #3 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/liamoreilly/Desktop
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (188) is hoisted out of
 the parallel loop labelled #3 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/lia
moreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (223)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (223) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        same_shape = (                                                     | 
            len(out_shape) == len(a_shape) == len(b_shape)                 | 
            and np.all(out_shape == a_shape)-------------------------------| #4
            and np.all(out_shape == b_shape)-------------------------------| #5
        )                                                                  | 
        same_strides = (                                                   | 
            len(out_strides) == len(a_strides) == len(b_strides)           | 
            and np.all(out_strides == a_strides)---------------------------| #6
            and np.all(out_strides == b_strides)---------------------------| #7
        )                                                                  | 
        # When `out`, `a`, `b` are stride-aligned, avoid indexing          | 
        if same_shape and same_strides:                                    | 
            for i in prange(len(out)):-------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(len(out)):-------------------------------------| #9
                out_index = np.empty(MAX_DIMS, dtype=np.int32)             | 
                a_index = np.empty(MAX_DIMS, dtype=np.int32)               | 
                b_index = np.empty(MAX_DIMS, dtype=np.int32)               | 
                to_index(i, out_shape, out_index)                          | 
                out_pos = index_to_position(out_index, out_strides)        | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                a_pos = index_to_position(a_index, a_strides)              | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                b_pos = index_to_position(b_index, b_strides)              | 
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/liamoreilly/Desktop
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (251) is hoisted out of
 the parallel loop labelled #9 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/liamoreilly/Desktop
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (252) is hoisted out of
 the parallel loop labelled #9 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/liamoreilly/Desktop
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (253) is hoisted out of
 the parallel loop labelled #9 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py 
(286)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (286) 
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   | 
        out: Storage,                                              | 
        out_shape: Shape,                                          | 
        out_strides: Strides,                                      | 
        a_storage: Storage,                                        | 
        a_shape: Shape,                                            | 
        a_strides: Strides,                                        | 
        reduce_dim: int,                                           | 
    ) -> None:                                                     | 
        # TODO: Implement for Task 3.1.                            | 
        for i in prange(len(out)):---------------------------------| #10
            out_index = np.empty(MAX_DIMS, np.int32)               | 
            size = a_shape[reduce_dim]                             | 
            to_index(i, out_shape, out_index)                      | 
            out_pos = index_to_position(out_index, out_strides)    | 
            a_pos = index_to_position(out_index, a_strides)        | 
            a = out[out_pos]                                       | 
            stride = a_strides[reduce_dim]                         | 
            for s in range(size):                                  | 
                a = fn(a, a_storage[a_pos])                        | 
                a_pos += stride                                    | 
            out[out_pos] = a                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/liamoreilly/Desktop
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (297) is hoisted out of
 the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/liam
oreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (312)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (312) 
------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                        | 
    out: Storage,                                                                   | 
    out_shape: Shape,                                                               | 
    out_strides: Strides,                                                           | 
    a_storage: Storage,                                                             | 
    a_shape: Shape,                                                                 | 
    a_strides: Strides,                                                             | 
    b_storage: Storage,                                                             | 
    b_shape: Shape,                                                                 | 
    b_strides: Strides,                                                             | 
) -> None:                                                                          | 
    """NUMBA tensor matrix multiply function.                                       | 
                                                                                    | 
    Should work for any tensor shapes that broadcast as long as                     | 
                                                                                    | 
    ```                                                                             | 
    assert a_shape[-1] == b_shape[-2]                                               | 
    ```                                                                             | 
                                                                                    | 
    Optimizations:                                                                  | 
                                                                                    | 
    * Outer loop in parallel                                                        | 
    * No index buffers or function calls                                            | 
    * Inner loop should have no global writes, 1 multiply.                          | 
                                                                                    | 
                                                                                    | 
    Args:                                                                           | 
    ----                                                                            | 
        out (Storage): storage for `out` tensor                                     | 
        out_shape (Shape): shape for `out` tensor                                   | 
        out_strides (Strides): strides for `out` tensor                             | 
        a_storage (Storage): storage for `a` tensor                                 | 
        a_shape (Shape): shape for `a` tensor                                       | 
        a_strides (Strides): strides for `a` tensor                                 | 
        b_storage (Storage): storage for `b` tensor                                 | 
        b_shape (Shape): shape for `b` tensor                                       | 
        b_strides (Strides): strides for `b` tensor                                 | 
                                                                                    | 
    Returns:                                                                        | 
    -------                                                                         | 
        None : Fills in `out`                                                       | 
                                                                                    | 
    """                                                                             | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                          | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                          | 
                                                                                    | 
    for i in prange(out_shape[0]):--------------------------------------------------| #13
        for j in prange(out_shape[1]):----------------------------------------------| #12
            for k in prange(out_shape[2]):------------------------------------------| #11
                a_inner = i * a_batch_stride + j * a_strides[1]                     | 
                b_inner = i * b_batch_stride + k * b_strides[2]                     | 
                acc = 0.0                                                           | 
                for _ in range(a_shape[2]):                                         | 
                    acc += a_storage[a_inner] * b_storage[b_inner]                  | 
                    a_inner += a_strides[2]                                         | 
                    b_inner += b_strides[1]                                         | 
                out_position = (                                                    | 
                    i * out_strides[0] + j * out_strides[1] + k * out_strides[2]    | 
                )                                                                   | 
                out[out_position] = acc                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None