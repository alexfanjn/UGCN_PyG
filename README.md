## UGCN_PyG

Reimplementation of NeurIPS 2021 paper "[Universal Graph Convolutional Networks](https://proceedings.neurips.cc/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html)" based on PyTorch and PyTorch Geometric (PyG).



## Run

```
python main.py
```



## Note

- Currently, we calculate the *k*NN neighbors by utilizing the package *sklearn*. As the Ball-tree implementation of *sklearn* does not support for directly using cosine similarity, we use the *'brute'* search instead. Therefore, our reimplementation for obtaining *k*NN neighbors here is not a linear algorithm.

  

