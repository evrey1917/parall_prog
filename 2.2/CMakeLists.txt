cmake_minimum_required(VERSION 3.8)

project(integra)

add_executable(${PROJECT_NAME} "./integral.c")
target_compile_options(integra PUBLIC "-fopenmp" "-lm" "-O2")
