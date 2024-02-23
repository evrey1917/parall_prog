cmake_minimum_required(VERSION 3.8)

project(multiply)

add_executable(${PROJECT_NAME} "./DGEMV.c")
target_compile_options(multiply PUBLIC "-fopenmp" "-lm" "-O2")
