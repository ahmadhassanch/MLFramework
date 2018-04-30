
Since I want to make a loss layer, it needs to take yRef as input, thus breaking
my assumption that we have only series connections in our network.

So ... the inputIndex and outputIndex stuff might be needed

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
- Make a loss layer
- write the Loss function
- Do the backward pass

- add error check in y dim.