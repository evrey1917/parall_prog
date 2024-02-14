cmake_minimum_required(VERSION 3.8)

project(sin_sum)

# float: ARRAY_TYPE = 0
# double: ARRAY_TYPE = 1
option(ARRAY_TYPE "Type of array elements: 0 = float (default); 1 = double" 0)

if(ARRAY_TYPE)
    add_compile_definitions(USE_DOUBLE)
    message("Array of DOUBLE")
else(ARRAY_TYPE)
    message("Array of FLOAT")
endif(ARRAY_TYPE)

add_executable(${PROJECT_NAME} "../sin_sum.cpp")
