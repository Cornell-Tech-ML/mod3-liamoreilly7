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
```
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
```
## 3.5 Results
### Simple
##### GPU time:
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
``` 
```

```
##### CPU time: 
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```

```
### Split
##### GPU time: 2mins -> 0.25s/epoch
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  6.089408831846174 correct 31
Epoch  10  loss  5.112833325097632 correct 39
Epoch  20  loss  4.6329430210413705 correct 46
Epoch  30  loss  5.784197596300064 correct 44
Epoch  40  loss  4.100355632672588 correct 46
Epoch  50  loss  1.9590212140148309 correct 49
Epoch  60  loss  2.7952550055178733 correct 49
Epoch  70  loss  0.860294094091502 correct 50
Epoch  80  loss  1.3526034281673023 correct 50
Epoch  90  loss  1.7757800280066438 correct 50
Epoch  100  loss  0.6926200572562278 correct 50
Epoch  110  loss  0.8253803416838955 correct 49
Epoch  120  loss  1.2825825173413028 correct 50
Epoch  130  loss  0.33040031902946376 correct 49
Epoch  140  loss  0.8133865726780889 correct 50
Epoch  150  loss  1.0999600941998011 correct 49
Epoch  160  loss  0.7722325057990445 correct 50
Epoch  170  loss  0.4026658838103195 correct 50
Epoch  180  loss  0.4263914425976528 correct 50
Epoch  190  loss  0.6499838698413459 correct 50
Epoch  200  loss  0.3761631322138891 correct 50
Epoch  210  loss  0.5495441823238267 correct 50
Epoch  220  loss  0.7147039339668997 correct 50
Epoch  230  loss  0.20831260945799362 correct 50
Epoch  240  loss  0.12494165529879593 correct 50
Epoch  250  loss  0.37390044234246017 correct 50
Epoch  260  loss  0.11613346591517373 correct 50
Epoch  270  loss  0.13078957899267676 correct 50
Epoch  280  loss  0.20601038992521556 correct 50
Epoch  290  loss  0.27560036048698516 correct 50
Epoch  300  loss  0.23533515588534243 correct 50
Epoch  310  loss  0.6444174557298631 correct 50
Epoch  320  loss  0.20610767456316448 correct 50
Epoch  330  loss  0.7068046340853331 correct 50
Epoch  340  loss  0.5801739766186254 correct 50
Epoch  350  loss  0.07325540725641766 correct 50
Epoch  360  loss  0.5821250103288926 correct 50
Epoch  370  loss  0.08139715300035996 correct 50
Epoch  380  loss  0.40046208944649386 correct 50
Epoch  390  loss  0.036072822912131736 correct 50
Epoch  400  loss  0.06195277985197897 correct 50
Epoch  410  loss  0.06493159883220236 correct 50
Epoch  420  loss  0.6563974884538111 correct 50
Epoch  430  loss  0.23011746114093332 correct 50
Epoch  440  loss  0.014278888021675245 correct 50
Epoch  450  loss  0.08398166722582799 correct 50
Epoch  460  loss  0.12712013148487789 correct 50
Epoch  470  loss  0.21451479617237068 correct 50
Epoch  480  loss  0.1986953773430774 correct 50
Epoch  490  loss  0.09918765343962974 correct 50
```
##### CPU time 1 min -> 0.12s/epoch
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  9.765257889515988 correct 28
Epoch  10  loss  5.51455061406202 correct 38
Epoch  20  loss  5.402512940547736 correct 44
Epoch  30  loss  4.622359866444943 correct 40
Epoch  40  loss  4.39448439670828 correct 42
Epoch  50  loss  4.746484895234552 correct 48
Epoch  60  loss  1.971513169480819 correct 42
Epoch  70  loss  3.1056905320220936 correct 44
Epoch  80  loss  3.2927340366014928 correct 44
Epoch  90  loss  2.472572278336464 correct 48
Epoch  100  loss  1.6473912308378484 correct 46
Epoch  110  loss  1.6811313856059495 correct 48
Epoch  120  loss  0.7152641724554946 correct 49
Epoch  130  loss  1.8418782992387701 correct 46
Epoch  140  loss  0.9934325166747052 correct 48
Epoch  150  loss  1.5475998313221762 correct 47
Epoch  160  loss  1.7893396930670895 correct 49
Epoch  170  loss  0.6736086992145116 correct 49
Epoch  180  loss  1.5156016028599726 correct 49
Epoch  190  loss  2.297793106534926 correct 50
Epoch  200  loss  1.9268134734910185 correct 49
Epoch  210  loss  2.1678206905984223 correct 49
Epoch  220  loss  0.5988154872249927 correct 49
Epoch  230  loss  1.3214423357066947 correct 49
Epoch  240  loss  0.8699151523289403 correct 49
Epoch  250  loss  0.7961756286558129 correct 50
Epoch  260  loss  0.6245305917465022 correct 47
Epoch  270  loss  0.8456962448028718 correct 50
Epoch  280  loss  0.5522662298392422 correct 50
Epoch  290  loss  0.8701691259650997 correct 48
Epoch  300  loss  1.7269932389538316 correct 46
Epoch  310  loss  0.22351853899075513 correct 49
Epoch  320  loss  0.28550327102918194 correct 48
Epoch  330  loss  0.6203880756405353 correct 48
Epoch  340  loss  0.07174947671078402 correct 49
Epoch  350  loss  3.8093319070729206 correct 50
Epoch  360  loss  0.989918939625666 correct 50
Epoch  370  loss  0.9141960560880799 correct 49
Epoch  380  loss  0.2751608966358913 correct 49
Epoch  390  loss  0.4322221678404138 correct 50
Epoch  400  loss  0.8369008074479533 correct 50
Epoch  410  loss  0.10647732923435761 correct 50
Epoch  420  loss  0.36486746863535696 correct 50
Epoch  430  loss  0.0911490231278374 correct 47
Epoch  440  loss  0.9141115676983433 correct 49
Epoch  450  loss  0.5407985190426734 correct 50
Epoch  460  loss  0.7232722860028326 correct 49
Epoch  470  loss  1.3785150726776045 correct 48
Epoch  480  loss  0.568271800498606 correct 49
Epoch  490  loss  0.9366928134349591 correct 50
```
### Xor
##### GPU time: 15min -> 1.8s/epoch
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```
Epoch  0  loss  6.012496413341992 correct 35
Epoch  10  loss  4.667234101378654 correct 39
Epoch  20  loss  4.910599358212691 correct 42
Epoch  30  loss  3.4702587404714413 correct 42
Epoch  40  loss  4.093085162433748 correct 45
Epoch  50  loss  3.7763875603340127 correct 45
Epoch  60  loss  2.269177619338483 correct 46
Epoch  70  loss  1.6276216585319836 correct 45
Epoch  80  loss  1.4586911307577095 correct 47
Epoch  90  loss  2.025229565446711 correct 46
Epoch  100  loss  2.1408797197109766 correct 48
Epoch  110  loss  1.8239860333826898 correct 49
Epoch  120  loss  3.3954847836031674 correct 44
Epoch  130  loss  1.9868966478918697 correct 48
Epoch  140  loss  1.7618435221407114 correct 48
Epoch  150  loss  2.0785551340950295 correct 45
Epoch  160  loss  1.4423182421167302 correct 48
Epoch  170  loss  1.6803931771085123 correct 48
Epoch  180  loss  1.3009994541457006 correct 48
Epoch  190  loss  1.8393835179879916 correct 49
Epoch  200  loss  1.13924987721975 correct 48
Epoch  210  loss  1.6802550598344448 correct 49
Epoch  220  loss  0.7437170760328987 correct 49
Epoch  230  loss  0.6906651720596862 correct 48
Epoch  240  loss  2.683718894407389 correct 49
Epoch  250  loss  0.3912989878337717 correct 49
Epoch  260  loss  0.8338760857063754 correct 48
Epoch  270  loss  0.7883855947081948 correct 46
Epoch  280  loss  0.8130317603903228 correct 48
Epoch  290  loss  0.778465621275914 correct 47
Epoch  300  loss  1.0747821901468 correct 48
Epoch  310  loss  0.609386915305761 correct 48
Epoch  320  loss  0.12052451370712249 correct 46
Epoch  330  loss  0.13502395897654634 correct 48
Epoch  340  loss  0.788542979266913 correct 49
Epoch  350  loss  2.0531081601213645 correct 50
Epoch  360  loss  0.29887883488391437 correct 48
Epoch  370  loss  0.4376361819077048 correct 50
Epoch  380  loss  1.776609843442601 correct 47
Epoch  390  loss  0.08250594507772245 correct 48
Epoch  400  loss  0.3480359469558896 correct 50
Epoch  410  loss  0.08239905132857857 correct 49
Epoch  420  loss  0.9977820896883873 correct 50
Epoch  430  loss  0.3106114998532404 correct 50
Epoch  440  loss  1.1131020207440168 correct 50
Epoch  450  loss  2.024612441404877 correct 48
Epoch  460  loss  1.5137310371314068 correct 48
Epoch  470  loss  1.3194082746648284 correct 47
Epoch  480  loss  0.42367435441467344 correct 50
Epoch  490  loss  0.4693781180201252 correct 49
```
##### CPU time: 1min -> 0.12s/epoch
```
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```
Epoch  0  loss  6.092275621529023 correct 31
Epoch  10  loss  4.698587654693932 correct 36
Epoch  20  loss  4.774508860040029 correct 45
Epoch  30  loss  4.092628823327241 correct 47
Epoch  40  loss  4.162846669191462 correct 46
Epoch  50  loss  3.4505080630861125 correct 48
Epoch  60  loss  3.1707704142264004 correct 46
Epoch  70  loss  3.6070864144932138 correct 47
Epoch  80  loss  2.221865839874837 correct 49
Epoch  90  loss  1.6621235524665778 correct 48
Epoch  100  loss  1.048656856276127 correct 48
Epoch  110  loss  0.9696712795237397 correct 49
Epoch  120  loss  1.0232139468045647 correct 50
Epoch  130  loss  1.351765333979722 correct 47
Epoch  140  loss  1.7110301077739998 correct 48
Epoch  150  loss  0.9057838290163296 correct 49
Epoch  160  loss  0.4097867477995856 correct 50
Epoch  170  loss  1.2015593552680053 correct 49
Epoch  180  loss  0.8556058810452615 correct 49
Epoch  190  loss  1.8625036369098944 correct 47
Epoch  200  loss  0.5571529118406311 correct 50
Epoch  210  loss  1.2898050994667714 correct 50
Epoch  220  loss  1.0125623260183412 correct 50
Epoch  230  loss  1.1596098118262093 correct 50
Epoch  240  loss  0.9176713866428919 correct 50
Epoch  250  loss  1.258613221421657 correct 50
Epoch  260  loss  0.5425782539708122 correct 50
Epoch  270  loss  0.5243254621783178 correct 50
Epoch  280  loss  0.161772470296409 correct 50
Epoch  290  loss  0.5098678588892158 correct 50
Epoch  300  loss  0.8883550647945259 correct 50
Epoch  310  loss  0.4136725536858709 correct 50
Epoch  320  loss  0.8445289913527879 correct 50
Epoch  330  loss  0.7604441425675198 correct 50
Epoch  340  loss  0.7472478654351566 correct 50
Epoch  350  loss  0.30504620056384624 correct 50
Epoch  360  loss  0.6253948602116594 correct 50
Epoch  370  loss  0.2949941041064886 correct 50
Epoch  380  loss  0.7926500214344898 correct 50
Epoch  390  loss  0.08745965905806485 correct 50
Epoch  400  loss  0.12851619677577159 correct 50
Epoch  410  loss  0.2400588477474389 correct 50
Epoch  420  loss  0.3143089611599509 correct 50
Epoch  430  loss  0.3970675538939845 correct 50
Epoch  440  loss  0.4832469607812148 correct 50
Epoch  450  loss  0.4887771289656861 correct 50
Epoch  460  loss  0.43140815023340817 correct 50
Epoch  470  loss  0.5367923365474838 correct 50
Epoch  480  loss  0.3514513628283146 correct 50
Epoch  490  loss  0.11728529591504278 correct 50
```