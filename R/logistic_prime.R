logistic_prime <-
function(v, s = 1){
  logistic(v, s) * (1- logistic(v, s))
}
