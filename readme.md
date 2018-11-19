# A simple kmeans algorithm with minimal dependency
* only depends on STL
* written in the style of STL algorithms
* works with most STL containers
## Requirements
* a C++14 compiler (tested with gcc 8.2) & STL
* containers that provide iterators (ideally, RandomAccess iterators); `std::vector` and `std::array` both work
* custom iterators should provide traits (`value_type` at the very least)
* numeric types that are closed under addition, and multipliable with floating point types
* CMake and GTest for unit test

## TODO
- [ ] Travis, coverage
- [ ] Parellel algorithm (AppVeyor Windows build?)
- [ ] CUDA
- [ ] test with SequentialAccess containers for fun
