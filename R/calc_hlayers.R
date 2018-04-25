
calc_hlayers <- function(parlist, X = X, param = param, fe_var = fe_var, nlayers = nlayers, convolutional, activation){
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
  if (length(nlayers) >1){
    hlayers <- foreach(i = 1:length(nlayers)) %do% {
      vector('list', nlayers[i])
    }
    names(hlayers) <- paste0("p", 1:length(hlayers))
    for (p in 1:length(nlayers)){
      for (i in 1:(nlayers[p])){
        if (i == 1){D <- X[[p]]} else {D <- hlayers[[p]][[i-1]]}
        D <- as.matrix(cbind(1, D)) #add bias
        hlayers[[p]][[i]] <- activ(MatMult(D, parlist[[p]][[i]]))        
      }
      colnames(hlayers[[p]][[i]]) <- paste0("p_",p, '_nodes',1:ncol(hlayers[[p]][[i]]))
    }
    hlayers$param <- param
    colnames(hlayers$param) <- paste0('param',1:ncol(param))
    if (is.null(fe_var)){#add intercept if no FEs
      hlayers$param <- cbind(1, hlayers$param)
    }
    return(hlayers)
  } else {
    hlayers <- vector('list', nlayers)
    for (i in 1:(nlayers + !is.null(convolutional))){
      if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
      D <- cbind(1, D) #add bias
      # make sure that the time-invariant variables pass through the convolutional layer without being activated
      if (is.null(convolutional) | i > 1){
        hlayers[[i]] <- activ(MatMult(D, parlist[[i]]))        
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
}
