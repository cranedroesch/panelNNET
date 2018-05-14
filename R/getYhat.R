# function to compute fitted values, potentially for subsets of the data

getYhat <- function(pl, hlay, param, y, ydm, fe_var, nlayers){ 
  if (!is.null(fe_var)){
    Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
      hlay[[i]][[length(hlay[[i]])]]
    }
    Z <- cbind(param, as.matrix(Z))
    Zdm <- demeanlist(Z, list(fe_var))
    B <- foreach(i = 1:length(nlayers), .combine = c) %do% {pl[[i]]$beta}
    fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(pl$beta_param, B)))
    yhat <- MatMult(Z, c(pl$beta_param, B)) + fe    
  } else {
    stop("not implemented yet for no FE's")
  }            
  return(as.numeric(yhat))
}

# old version

# getYhat <- function(pl, hlay = NULL){ 
#   #Update hidden layers
#   if (is.null(hlay)){hlay <- calc_hlayers(pl,
#                                           X = X,
#                                           param = param,
#                                           fe_var = fe_var,
#                                           nlayers = nlayers,
#                                           convolutional = convolutional,
#                                           activ = activation)}
#   #update yhat
#   if (length(nlayers)>1){
#     if (!is.null(fe_var)){
#       Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
#         hlay[[i]][[length(hlay[[i]])]]
#       }
#       Z <- cbind(param, as.matrix(Z))
#       Zdm <- demeanlist(Z, list(fe_var))
#       B <- foreach(i = 1:length(nlayers), .combine = c) %do% {parlist[[i]]$beta}
#       fe <- (y-ydm) - MatMult(as.matrix(Z)-Zdm, as.matrix(c(pl$beta_param, B)))
#       yhat <- MatMult(Z, c(pl$beta_param, B)) + fe    
#     } else {
#       stop("not implement for no FE's")
#     }            
#   } else {
#     if (!is.null(fe_var)){
#       Zdm <- demeanlist(as.matrix(hlay[[length(hlay)]]), list(fe_var))
#       fe <- (y-ydm) - MatMult(as.matrix(hlay[[length(hlay)]])-Zdm, as.matrix(c(pl$beta_param, pl$beta)))
#       yhat <- MatMult(hlay[[length(hlay)]], c(pl$beta_param, pl$beta)) + fe    
#     } else {
#       yhat <- MatMult(hlay[[length(hlay)]], c(pl$beta_param, pl$beta))
#     }      
#   }
#   return(as.numeric(yhat))
# }

