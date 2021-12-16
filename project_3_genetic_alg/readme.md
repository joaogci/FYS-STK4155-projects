
# Differential Equation Genetic Algorithm solver

The implementation presented here is in part adapted from [this paper by I. G. Tsoulos & I. E. Lagaris](http://dx.doi.org/10.1007/s10710-006-7009-y).

## Getting started

The project can be opened and built as is with Visual Studio 2019 on Windows, or compiled with gcc through `make` (C++14).

## Todo

- Try alternative chromosome representation with chromosomes being the expression trees themselves instead of decodable sequences of integers - genetic rules will need to be modified accordingly, to:
    - A probability of each individual node being modified (+ -> -, 5 -> 5.4, sin -> cos, etc.)
    - A probability of a sub-tree being replaced with a new random sub-tree/expression (e.g. sin(5x + 6) -> sin(5x + exp(-1/x)))
    - A probability of nesting a sub-tree inside another expression (e.g. sin(5x + 6) -> sin(cos(5x + 6)), or sin(5x + 6) -> 1.2sin(5x + 6))
    - Crossovers by sub-trees/expressions instead of raw genes
