
calc_hlayers <- function(parlist, 
                         X = X, 
                         param = param, 
                         cs_var = cs_var, 
                         nlayers = nlayers, 
                         convolutional, 
                         activation,
                         effects){
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
  if (is.null(cs_var) | effects == "random"){#add intercept if no FEs or if using random effects
    hlayers[[i]] <- cbind(1, hlayers[[i]])
    colnames(hlayers[[i]])[1] <- "intercept"
  }
  return(hlayers)
}
