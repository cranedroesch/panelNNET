

Jacobian.panelNNET <- function(obj, numerical = FALSE, parallel = TRUE
  , step = 1e-9, newX = NULL, new.param = NULL, fe.newX = NULL, ...){
  if (numerical == FALSE){
    Jacobian.predictmethod(obj = obj, parallel = parallel, step = step
      , newX = newX, new.param = new.param, fe.newX = fe.newX)
  } else {
    Jacobian.numerical(obj)
  }
}

Jacobian.numerical <- function(obj){
  if (obj$activation == 'tanh'){
    activ <- tanh
    activ_prime <- tanh_prime
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
    activ_prime <- logistic_prime
  }
  if (obj$activation == 'relu'){
    activ <- relu
    activ_prime <- relu_prime
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
    activ_prime <- lrelu_prime
  }
  plist <- as.relistable(obj$parlist)
  pvec <- unlist(plist)
  #define function to pass to `jacobian` from `numDeriv`
  Jfun <- function(pvec, obj){
    parlist <- relist(pvec)
    D <- obj$X
    for (i in 1:length(obj$hidden_units)){
      D <- cbind(1,D) #bias
      D <- activ(D %*% parlist[[i]])
    } 
    colnames(D) <- paste0('nodes',1:ncol(D))
    if (!is.null(obj$treatment)){
      #Add treatment interactions
      if (obj$interact_treatment == TRUE){
        ints <- sweep(D, 1, obj$treatment, '*')
        colnames(ints) <- paste0('TrInts',1:ncol(ints))
        D <- cbind(ints, D)
      }
      #Add treatment dummy
      D <- cbind(obj$treatment, D)
      colnames(D)[1] <- 'treatment'
    }
    if (!is.null(obj$param)){
      D <- cbind(obj$param, D)
      colnames(D)[1:ncol(obj$param)] <- paste0('param',1:ncol(obj$param))
    }
    if (is.null(obj$fe_var)){D <- cbind(1, D)}#add intercept if no FEs
    if (!is.null(obj$fe_var)){
      Zdm <- demeanlist(D, list(obj$fe_var))
      ydm <<- demeanlist(obj$y, list(obj$fe_var))
      fe <- (obj$y-ydm) - as.matrix(D-Zdm) %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
      yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta) + fe    
    } else {
      yhat <- D %*% c(parlist$beta_param, parlist$beta_treatment, parlist$beta_treatmentinteractions, parlist$beta)
    }
    return(yhat)
  }
  #pass `Jfun` to `jacobian` from `numDeriv`
  J <- jacobian(Jfun, pvec, obj = obj)
  #drop any zero columns that represent lower-level parameters
  dJ <- ncol(J)
  tokeep <- which(!(apply(J, 2, function(x){all(x==0)}) & !grepl('treatment|param', names(pvec))))
  J <- J[,tokeep]
  if (ncol(J) < dJ){
    warning(paste0(dJ - ncol(J), ' columns dropped from Jacobian because dY/dParm =~ 0'))
  }
  pvec <- pvec[tokeep]
  #re-order jacobian so that parametric terms are on the front, followed by top layer.
  J <- J[,c(
      which(grepl('param', names(pvec)))
    , which(names(pvec) == 'beta_treatment')
    , which(grepl('beta_treatmentinteractions', names(pvec)))
    , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)) & !grepl('treatment', names(pvec)))
    , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
   )]
  return(J)
}


Jacobian.predictmethod <- function(obj, parallel, step, newX, new.param, fe.newX){
#parallel <- TRUE
#obj <- pnn
#step = 1e-12
  `%fun%` <- ifelse(parallel == TRUE, `%dopar%`, `%do%`)
  pvec <- unlist(obj$parlist)  
  J <- foreach(i = 1:length(pvec), .combine = cbind) %fun% {
    pv1 <- pv2 <- pvec
    pv1[i] <- pv1[i] + step
    pv2[i] <- pv2[i] - step
    obj1 <- panelNNET(y = obj$y, X = obj$X, hidden_units = obj$hidden_units, fe_var = obj$fe_var
      , maxit = 0, lam = obj$lam, param = obj$param, activation = obj$activation
      , inference = FALSE
      , parlist = relist(pv1), parapen = 0
    )
    obj2 <- panelNNET(y = obj$y, X = obj$X, hidden_units = obj$hidden_units, fe_var = obj$fe_var
      , maxit = 0, lam = obj$lam, param = obj$param, activation = obj$activation
      , inference = FALSE
      , parlist = relist(pv2), parapen = 0
    )
    #In-sample Jacobian
    if (is.null(newX)){
      dy1 <- (obj1$yhat - obj$yhat)/step
      dy2 <- (obj$yhat - obj2$yhat)/step
      dy <- rowMeans(dy1, dy2)
      dy <- matrix(dy)
      colnames(dy) <- names(pvec)[i]
      return(dy)
    } else {
      #unpeturbed object
      yhat <- predict(obj, newX = newX, new.param = new.param, fe.newX = fe.newX, se.fit = FALSE)
      #perturbed object
      yhat1 <- predict(obj1, newX = newX, new.param = new.param, fe.newX = fe.newX, se.fit = FALSE)
      yhat2 <- predict(obj2, newX = newX, new.param = new.param, fe.newX = fe.newX, se.fit = FALSE)
      #forward and backward differences
      dy1 <- (yhat1 - yhat)/step
      dy2 <- (yhat - yhat2)/step
      dy <- rowMeans(cbind(dy1, dy2))
      dy <- matrix(dy)
      colnames(dy) <- names(pvec)[i]
      return(dy)
    }
  }
  colnames(J) <- names(unlist(obj$parlist))
  J <- cbind(J[,grepl('beta', colnames(J))], J[,!grepl('beta', colnames(J))])
  return(J)
}


#Jacobian.analytical <- function(obj){

#  nlayers <- length(obj$hidden_layers)
#  bias_hlayers <- obj$used_bias
#  activation = obj$activation
#  X <- scale(obj$X)
#  param <- scale(obj$param)
#  hlayers <- obj$hidden_layers

#  getYhat <- function(pl, skel = attr(pl, 'skeleton'), hlay = NULL){ 
#    plist <- relist(pl, skel)
#    #Update hidden layers
#    if (is.null(hlay)){hlay <- calc_hlayers(plist)}
#    #update yhat
#    if (!is.null(fe_var)){
#      Zdm <- demeanlist(hlay[[length(hlay)]], list(fe_var))
#      fe <- (y-ydm) - as.matrix(hlay[[length(hlay)]]-Zdm) %*% as.matrix(c(
#          plist$beta_param, plist$beta_treatment
#        , plist$beta_treatmentinteractions, plist$beta
#      ))
#      yhat <- hlay[[length(hlay)]] %*% c(
#        plist$beta_param, plist$beta_treatment, plist$beta_treatmentinteractions, plist$beta
#      ) + fe    
#    } else {
#      yhat <- hlay[[length(hlay)]] %*% c(plist$beta_param, plist$beta_treatment, plist$beta_treatmentinteractions, plist$beta)
#    }
#    return(yhat)
#  }

#  lossfun <- function(pl, skel, lam, parapen){
#    yhat <- getYhat(pl, skel)
#    mse <- mean((y-yhat)^2)
#    plist <- relist(pl, skel)
#    loss <- mse + lam*sum(c(plist$beta_param*parapen, 0*plist$beta_treatment, plist$beta, plist$beta_treatmentinteractions, unlist(plist[!grepl('beta', names(plist))]))^2)
#    return(loss)
#  }

#  calc_hlayers <- function(parlist, normalize = FALSE){
#    hlayers <- vector('list', nlayers)
#    for (i in 1:nlayers){
#      if (i == 1){D <- X} else {D <- hlayers[[i-1]]}
#      if (bias_hlayers == TRUE){D <- cbind(1, D)}
#      if (normalize == TRUE){
#        hli <- activ(D %*% parlist[[i]])
#        v <- sd(as.numeric(hli))
#        hlayers[[i]] <- hli/v 
#      } else {
#        hlayers[[i]] <- activ(D %*% parlist[[i]])
#      }
#    }
#    colnames(hlayers[[i]]) <- paste0('nodes',1:ncol(hlayers[[i]]))
#    if (!is.null(param)){#Add parametric terms to top layer
#      hlayers[[i]] <- cbind(param, hlayers[[i]])
#      colnames(hlayers[[i]])[1:ncol(param)] <- paste0('param',1:ncol(param))
#    }
#    if (is.null(fe_var)){
#      hlayers[[i]] <- cbind(1, hlayers[[i]])#add intercept if no FEs
#    }
#    return(hlayers)
#  }

#  calc_grads<- function(plist, hlay = NULL, yhat = NULL, curBat = NULL){
##curBat = j
##hlay = lapply(obj$hidden_layers, function(x){x[j,,drop = FALSE]})
##yhat <- obj$yhat[j]
##plist <- obj$parlist
#    if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
#    if (is.null(hlay)){hlay <- calc_hlayers(plist)}
#    if (is.null(yhat)){yhat <- getYhat(unlist(plist), hlay = hlay)}
#    grads <- vector('list', nlayers+1)
#    grads[[length(grads)]] <- 1#getDelta(CB(as.matrix(y)), yhat)
#    for (i in (nlayers):1){
#      if (i == nlayers){outer_param = as.matrix(c(plist$beta))} else {outer_param = plist[[i+1]]}
#      if (i == 1){lay = CB(X)} else {lay= hlay[[i-1]]}
#      if (bias_hlayers == TRUE){
#        lay <- cbind(1, lay) #add bias to the hidden layer
#        if (i!=nlayers){outer_param <- outer_param[-1,]}      #remove parameter on upper-layer bias term
#      }
#      grads[[i]] <- getS(D_layer = lay, inner_param = plist[[i]], outer_deriv = grads[[i+1]], outer_param = outer_param, activation)
#    }
#    return(grads)
#  }


#  J <- foreach(j = 1:nrow(X), .combine = rbind) %do% {
#    grads <- calc_grads(obj$parlist, lapply(obj$hidden_layers, function(x){x[j,,drop = FALSE]}), obj$yhat[j], curBat = j)
#    gr <- foreach(i = 1:(length(hlayers)+1)) %do% {
#      if (i == 1){D <- X[j,]} else {D <- hlayers[[i-1]][j,]}
#      if (bias_hlayers == TRUE & i != length(hlayers)+1){D <- c(1, D)}
#        (t(t(D)) %*% grads[[i]])
#    }
#    unlist(gr)
#  }
#  B <- (ncol(J) - ncol(hlayers[[length(hlayers)]])+1):ncol(J)
#  J <- cbind(J[,B], J[,1:min(B-1)])
#  return(J)
#}




#Jacobian.analytical <- function(obj){
#  if (obj$activation == 'tanh'){
#    activ <- tanh
#    activ_prime <- tanh_prime
#  }
#  if (obj$activation == 'logistic'){
#    activ <- logistic
#    activ_prime <- logistic_prime
#  }
#  if (obj$activation == 'relu'){
#    activ <- relu
#    activ_prime <- relu_prime
#  }
#  if (obj$activation == 'lrelu'){
#    activ <- lrelu
#    activ_prime <- lrelu_prime
#  }
#  #Start parameter data frame
#  par.df <- data.frame(par = unlist(obj$parlist))
#  #make list of only lower parameters
#  lowerpars <- obj$parlist
#  lowerpars$beta <- lowerpars$beta_param <- NULL
#  #add variable signifying their level
#  lev <- foreach(i = 1:length(lowerpars), .combine = c) %do% {
#    rep(i, length(lowerpars[[i]]))
#  }
#  #add top-level parameters
#  par.df$lev <- c(lev, rep('beta_param', length(obj$parlist$beta_param)), rep('beta', length(obj$parlist$beta)))
#  #Which column of their layer they multiply
#  which.layer <- foreach(i = 1:length(lowerpars), .combine = rbind) %do% {
#    foreach(j = 1:ncol(lowerpars[[i]]), .combine = rbind) %do% {
#      lev.pointer <- foreach(k = (1+obj$used_bias):nrow(lowerpars[[i]]), .combine = c) %do% {
#        k
#      }
#      upper.lev.pointer <- foreach(k = (obj$used_bias):nrow(lowerpars[[i]]), .combine = c) %do% {
#        j+ifelse(i == length(lowerpars) & !is.null(obj$fe_var), 0, obj$used_bias) 
#        #^No bias at top later if we're dealing with a fixed-effects model
#      }
#      if(obj$used_bias){
#        lev.pointer <- lev.pointer - 1
#        lev.pointer <- c('bias', lev.pointer)
#      }
#      cbind(lev.pointer, upper.lev.pointer)
#    }
#  }
#  par.df$lev.pointer <- c(which.layer[,1], rep(NA, length(c(obj$parlist$beta, obj$parlist$beta_param))))
#  par.df$upper.lev.pointer <- c(which.layer[,2], rep(NA, length(c(obj$parlist$beta, obj$parlist$beta_param))))
#  #^the betas in fact emanate from a level, but their jacobins are invariant to this
#  #use chain rule to get a list of matrices of the form 
#  #a'(V_L B_L)B_L-1.  
#  chains <- foreach(L = 0:(length(obj$hidden_layers)-1)) %do% {
#    bias <- ifelse(obj$used_bias, 1, NULL)
#    if (L == 0){
#      D <- cbind(bias, obj$X)
#    } else {
#      if (L != length(obj$hidden_layers)){
#        D <- cbind(bias, obj$hidden_layers[[L]])
#      } else { #top layer shouldn't include parametric terms in ere
#        D <- obj$hidden_layers[[L]]
#        D <- D[,grepl('nodes', colnames(D))]
#        D <- cbind(bias, obj$hidden_layers[[L]])
#      }
#    }
#    lower_par <- obj$parlist[[L+1]]
#    if ((L+2) > length(obj$hidden_layers)){
#      upper_par <- obj$parlist$beta #upper level param will be beta at the top level
#    } else {
#      upper_par <- obj$parlist[[L+2]]
#    }
#    #a'(layer)
#    if ((L+2) > length(obj$hidden_layers)){
#      chn <- activ_prime(D %*% lower_par) %*% upper_par
#    } else {
#      chn <- cbind(bias, activ_prime(D %*% lower_par)) %*% upper_par
#    }
#    return(chn)
#  }
#  chains <- lapply(chains, rowSums)#they will only get used summed, so sum them now
#  #Lev is the level that the parameter emanates from
#  #lev.pointer is the column of the level that the parameter emanates from
#  #upper.lev.pointer is the column of the upper level that the parameter goes to
#  #Note that the bias terms are not included in the saved hidden layers, nor in the X.  So sometimes the number one needs to be added or subtracted from the index
#  Jacobian_ab <- foreach(i = 1:nrow(par.df[!grepl('beta', par.df$lev),]), .combine = cbind) %do% {
#    #Get vector of (derived) variable at current level
#    if (par.df$lev[i] == "1"){
#      if (par.df$lev.pointer[i] == 'bias') {
#        lay <- rep(1, nrow(obj$X))
#      } else {
#        lay <- obj$X[,as.numeric(par.df$lev.pointer[i])]
#      }
#    } else {
#      if (par.df$lev.pointer[i] == 'bias') {
#        lay <- rep(1, nrow(obj$X))
#      } else {
#        lay <- obj$hidden_layers[[as.numeric(par.df$lev[i])-1]][,as.numeric(par.df$lev.pointer[i])]
#      }  
#    }
#    #Start calculating Jacobia column.  At the top level, we're just dealing with the layer
#    Jcol <- lay
#    if (!is.na(par.df$upper.lev.pointer[i])){
#      #The layer directly above is the row-wise sum of the single linked layer and the paramters emanating from it
#      if (class(obj$parlist[[as.numeric(par.df$lev[i])+1]]) == 'numeric'){#matrix or vector determines which layer we're on
#        above.pars <- obj$parlist$beta[as.numeric(par.df$upper.lev.pointer[i])]
#        above.layer <- obj$hidden_layers[[as.numeric(par.df$lev[i])]][,as.numeric(par.df$upper.lev.pointer[i])+ncol(obj$param)]
#      } else {
#        above.pars <- obj$parlist[[as.numeric(par.df$lev[i])+1]][as.numeric(par.df$upper.lev.pointer[i]),]
#        above.layer <- obj$hidden_layers[[as.numeric(par.df$lev[i])]][,as.numeric(par.df$upper.lev.pointer[i])-1]
#      }
#      above <- rowSums(as.matrix(activ_prime(above.layer)) %*% t(as.matrix(above.pars)))
#      Jcol <- Jcol * above
#      #After the top layer, matrix expressions work again
#      #Element-wise multiply the row-sums of the chains for the corresponding upper layers
#      if (as.numeric(par.df$lev[i]) < length(lowerpars) & !is.na(as.numeric(par.df$lev[i]))){
#        chainstart <- as.numeric(par.df$lev[i]) + 1 
#        top <- foreach(ch = chainstart:length(chains), combine = '*') %do% {
#          chains[[ch]]
#        }
#        top <- unlist(top)

#        #Calculate jacobin column
#        Jcol <- top * above * lay
#      }
#    }
#    return(Jcol)
#  }
#  Jnames <- paste0('l',par.df$lev,'f',par.df$lev.pointer,'t',par.df$upper.lev.pointer)
#  colnames(Jacobian_ab) <- Jnames[!grepl('beta',Jnames)]
#  #Bind on the top layer such that the parametric terms are at the front
#  toplayer <- obj$hidden_layers[[length(obj$hidden_layers)]]
#  scaled_parametric <- toplayer[,grepl('param', colnames(toplayer))]
#  visible_layer <- toplayer[,grepl('nodes', colnames(toplayer))]
#  Jacobian <- cbind(scaled_parametric, visible_layer, Jacobian_ab)
#  return(Jacobian)
#}




