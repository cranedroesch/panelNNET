
calc_hlayers <- function(parlist, BNparm, X = X, param = param, fe_var = fe_var, nlayers = nlayers, convolutional, activation){
  if (activation == 'tanh'){
    activ <- tanh
  }
  if (activation == 'logistic'){
    activ <- logistic
  }
  if (activation == 'relu'){
    activ <- relu
  }
  if (activation == 'lrelu'){
    activ <- lrelu
  }
  hlayers <- vector('list', nlayers)
  # define re-used variables
  N <- nrow(X)
  lBN <- unlist(sapply(lapply(parlist, dim), function(x){unlist(x)[2]}))
  # loop through
  for (i in 1:(nlayers + !is.null(convolutional))){
    if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
    D <- cbind(1, D) #add bias
    # make sure that the time-invariant variables pass through the convolutional layer without being activated
    if (is.null(convolutional) | i > 1){
      if (is.null(BNparm)){
        hlayers[[i]] <- activ(t(matrix(rep(BNparm[[i]]$G, N), lBN[i])) *
                                colScale(MatMult(D, parlist[[i]]), add_attr = FALSE) +
                                t(matrix(rep(BNparm[[i]]$B, N), lBN[i])))
      } else {
        hlayers[[i]] <- activ(MatMult(D, parlist[[i]])) 
      }
    } else {
      HL <- MatMult(D, parlist[[i]])
      HL[,1:(convolutional$N_TV_layers * convolutional$Nconv)] <- activ(HL[,1:(convolutional$N_TV_layers * convolutional$Nconv)])
      hlayers[[i]] <- HL
    }
  }
  colnames(hlayers[[i]]) <- paste0('nodes',1:ncol(hlayers[[i]]))
  if (!is.null(param)){#Add parametric terms to top layer
    hlayers[[i]] <- cbind(param, hlayers[[i]])
    colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
  }
  if (is.null(fe_var)){#add intercept if no FEs
    hlayers[[i]] <- cbind(1, hlayers[[i]])
  }
  return(hlayers)
}
