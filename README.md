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

## Task 3.1 Diagnostics Output

MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/lia
moreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (163)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        # TODO: Implement for Task 3.1.                                      |
        out_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------| #0
        in_index: Index = np.zeros(MAX_DIMS, np.int32)-----------------------| #1
        if (                                                                 |
            len(out_strides) != len(in_strides)                              |
            or (out_strides != in_strides).any()-----------------------------| #2
            or (out_shape != in_shape).any()---------------------------------| #3
        ):                                                                   |
            for i in prange(len(out)):---------------------------------------| #5
                to_index(i, out_shape, out_index)                            |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                in_pos = index_to_position(in_index, in_strides)             |
                out_pos = index_to_position(out_index, out_strides)          |
                out[out_pos] = fn(in_storage[in_pos])                        |
        else:                                                                |
            # When `out` and `in` are stride-aligned, avoid indexing         |
            for i in prange(len(out)):---------------------------------------| #4
                out[i] = fn(float(in_storage[i]))                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #0, #2, #3, #5, #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--0 (parallel)
+--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--0 (parallel, fused with loop(s): 1)



Parallel region 0 (loop #0) had 1 loop(s) fused.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/lia
moreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (216)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (216)
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
        out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------------| #6
        a_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------| #7
        b_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------| #8
        if (                                                               |
            len(out_strides) != len(a_strides)                             |
            or len(out_strides) != len(b_strides)                          |
            or (out_strides != a_strides).any()----------------------------| #9
            or (out_strides != b_strides).any()----------------------------| #10
            or (out_shape != a_shape).any()--------------------------------| #11
            or (out_shape != b_shape).any()--------------------------------| #12
        ):                                                                 |
            for i in prange(len(out)):-------------------------------------| #14
                to_index(i, out_shape, out_index)                          |
                broadcast_index(out_index, out_shape, a_shape, a_index)    |
                broadcast_index(out_index, out_shape, b_shape, b_index)    |
                a_pos = index_to_position(a_index, a_strides)              |
                b_pos = index_to_position(b_index, b_strides)              |
                out_pos = index_to_position(out_index, out_strides)        |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      |
        else:                                                              |
            # When out, a, b are stride-aligned, avoid indexing            |
            for i in prange(len(out)):-------------------------------------| #13
                out[i] = fn(a_storage[i], b_storage[i])                    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--6 has the following loops fused into it:
   +--7 (fused)
   +--8 (fused)
Following the attempted fusion of parallel for-loops there are 7 parallel for-
loop(s) (originating from loops labelled: #6, #9, #10, #11, #12, #14, #13).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
+--7 (parallel)
+--8 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel, fused with loop(s): 7, 8)



Parallel region 0 (loop #6) had 2 loop(s) fused.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py
(276)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/liamoreilly/Desktop/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (276)
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
        for i in prange(len(out)):---------------------------------| #15
            out_index = np.empty(MAX_DIMS, np.int32)               |
            size = a_shape[reduce_dim]  # the reduce size          |
            to_index(i, out_shape, out_index)                      |
                                                                   |
            out_pos = index_to_position(out_index, out_strides)    |
            a_pos = index_to_position(out_index, a_strides)        |
            acc = out[out_pos]                                     |
            step = a_strides[reduce_dim]                           |
            for s in range(size):                                  |
                acc = fn(acc, a_storage[a_pos])                    |
                a_pos += step                                      |
            out[out_pos] = acc                                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #15).
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
/CornellTech/MLE/mod3-liamoreilly7/minitorch/fast_ops.py (287) is hoisted out of
 the parallel loop labelled #15 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.