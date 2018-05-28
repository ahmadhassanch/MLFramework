# MLFramework
A Deep Machine Learning framework.


- be able to access layer by layername
- remove nnData
- print percentage correct results at each iteration
              Actual   Model   Result
  correct       50      90       xx
  incorrect     50      10       xx

- basic test that we should have all types of samples
  positive and negative (and in reasonable ratio)


Questions:
- Why all positive initializations
- See why output is 0.693
- why only b changes in case of incorrect initialization
  (the weights are small and output decays)
  by the time output reaches sigmoid, it is zero, so output 
  of sigmoid is 1/(1+1)=0.5
  > normalizing the output should fix it.