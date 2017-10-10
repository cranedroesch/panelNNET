# panelNNET
Semiparametric panel data models using neural networks

TBD:

1.  Update manpages

2.  Need a method for including groups that aren't necessarily represented by fixed effects in estimating cluster vcv

3.  GPU integration

4.  Add effective degrees of freedom to summary output

5.  Build interactive mode, using the keypress package

6.  Write a vignette

7.  Save activations as functions, rather than strings/pointers, then remove all of the redundant headers in the various files

8.  Remove storage of hidden layers to degree possible, to reduce memory footprint.

9.  Reduce number of things in the output, perhaps subject to an argument.  Goal is to reduce storage footprint and loading time.  This will involve not storing the input data, but storing the scaling factors from the input data.

10.  Convolutional throws an error when there are no fixed variables.  This is because of the way the convmask building function binds the time-varying and non-time-varying portions of the mask together -- it assumes that there is a non-time-varying portion.

11.  Speed up the calc_grads function

12.  Speed up the OLStrick.

13.  Get dropout to work with convolutional nets
