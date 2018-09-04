
# function to compute fitted values, potentially for subsets of the data
getYhat <- function(pl, hlay, param, y, ydm, fe_var, nlayers, weights){ 
  Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
    hlay[[i]][[length(hlay[[i]])]]
  }
  if (!is.null(fe_var)){
    Z <- cbind(param, as.matrix(Z))
    Zdm <- demeanlist(Z, list(fe_var), threads = 1, weights = weights^.5)
    B <- foreach(i = 1:length(nlayers), .combine = c) %do% {pl[[i]]$beta}
    fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(pl$beta_param, B)))
    yhat <- MatMult(Z, c(pl$beta_param, B)) + fe
  } else {
    Z <- cbind(hlay$param, as.matrix(Z)) # `hlay$param` should be a vector of ones
    B <- foreach(i = 1:length(nlayers), .combine = c) %do% {pl[[i]]$beta}
    B <- c(pl$beta_param, B)
    yhat <- MatMult(Z, B)
  }            
  return(as.numeric(yhat))
}

