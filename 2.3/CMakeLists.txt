cmake_minimum_required(VERSION 3.8)

project(iter_var1)
project(iter_var2)

add_executable(iter_var1 "./ir_var1.c")
add_executable(iter_var2 "./ir_var2.c")
target_compile_options(iter_var1 PUBLIC "-fopenmp" "-lm" "-O2")
target_compile_options(iter_var2 PUBLIC "-fopenmp" "-lm" "-O2")
