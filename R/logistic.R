logistic <-
function(v, s = 1){
  1/(1+exp(-v*s))
}
