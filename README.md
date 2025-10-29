# PV197
GPU programming project

This is my project for the course PV197. The project description is as follows:

The goal of the project is to implement an GPU-accelerated code
that compute a moving average of an input vector
 - the input is a vector in of n floating point numbers
 - the output is a vector out containing moving average of the input
 - for an element outi , the average is computed as: \frac{\sum_{j=i-R}^{i+R-1}}{2R} 
 - where R is half of the average window
 - when in is accessed out of bound, the first or last element is read

What will be tested?
 - the input size predefined in framework.cu can be changed (can be rectangular)
 - the input/output size will be divisible by 1024
 - the R size will be power of two, and R ≤ 1024
 - the code have to run on computing capability 5.0 and newer

What is forbidden?
 - using external libraries (write kernel code)
 - collaboration (discuss general questions, not your code)
 - using chatbots is not recommended, especially in early stage of the project

Tested on:
 - "NVIDIA GeForce RTX 2080 Ti"
