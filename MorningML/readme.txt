
Done:
- Introduced input layer
- Now we only define the number of 'output' units in layer only.
- WX + B implemented for inner product
- 1/(1+exp(-z)) for the sigmoid layer
- Relu implemented in relu layer
- NNActivation layer introduced as parent of activation layers
  > to check the consistency in dimension of input layer
- The loop simplified by (for layer in self.layers:)
To do:
- do the sigmoid for sigma/relu, etc.
- write the Loss function
- Do the backward pass

