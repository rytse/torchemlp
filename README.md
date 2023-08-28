# torchemlp
Torch implementation of [Marc Finzi's Equivariant MLP](https://github.com/mfinzi/equivariant-MLP).

## Tools

### Pyright

Before running [pyright](https://github.com/microsoft/pyright), generate the missing type stubs for libraries with

```
$ pyright --createstub matplotlib
$ pyright --createstub functorch
```

This will generate type stubs in the gitignored `typings/` directory. After this step, you can run static analysis on the entire `torchemlp` library with

```
$ pyright
```
