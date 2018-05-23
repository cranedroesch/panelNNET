



predict.panelNNET <-
function(obj, y_test = NULL, newX = NULL, fe.newX = NULL, new.param = NULL, se.fit = FALSE
         , numerical_jacobian = FALSE, parallel_jacobian = FALSE, convolutional = NULL){
# obj = pr_obj
# y_test = stop_early$y_test
# newX = stop_early$X_test
# fe.newX = stop_early$fe_test
# new.param = stop_early$P_test
  if (obj$activation == 'tanh'){
    activ <- tanh
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
  }
  if (obj$activation == 'relu'){
    activ <- relu
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
  }
  if (is.null(newX)){
    return(obj$yhat)
  } else {
    if (class(newX) != "list"){
      newX <- list(newX)
    }
    if (!all(unique(fe.newX) %in% unique(obj$fe$fe_var)) & is.null(y_test)){
      stop('New data has cross-sectional units not observed in training data')
    }
    #Scale the new data by the scaling rules in the training data
    plist <- as.relistable(obj$parlist) # pull out from list 
    pvec <- unlist(plist)
    #prepare fe's in advance...
    if (!is.null(obj$fe)){
      # The purpose of this is to correct small numerical differences between different CSU fixed effects
      FEs_to_merge <- summaryBy(fe~fe_var, data = obj$fe, keep.names = T)
      # If there is a labeled outcome in the test set (i.e.: early stopping) compute FE's and append them
      if (any(unique(fe.newX) %ni% unique(obj$fe$fe_var)) & !is.null(y_test)){
        # rescale new data to scale of training data
        D <- foreach(i = 1:length(obj$X)) %do% {
          sweep(sweep(newX[[i]][fe.newX %ni% obj$fe$fe_var,], 
                      2, 
                      STATS = attr(obj$X[[i]], "scaled:center"), FUN = '-'), 
                2, STATS = attr(obj$X[[i]], "scaled:scale"), FUN = '/')
        }
        if (!is.null(obj$param)){
          P <- sweep(sweep(new.param[fe.newX %ni% obj$fe$fe_var,], 
                           MARGIN = 2, 
                           STATS = attr(obj$param, "scaled:center"), 
                           FUN = '-'), 
                     MARGIN = 2, 
                     STATS = attr(obj$param, "scaled:scale"), 
                     FUN = '/')
        } else {P <- NULL}
        # compute hidden layers
        nlayers <- sapply(obj$hidden_layers, length)
        print(nlayers)
        # nlayers <- nlayers[!grepl("param", names(nlayers))]
        HL <- calc_hlayers(parlist = obj$parlist, 
                           X = D, 
                           param = P, 
                           fe_var = fe.newX[fe.newX %ni% obj$fe$fe_var], 
                           nlayers = nlayers,
                           activation = obj$activation)
        
        Z <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
          HL[[i]][[length(HL[[i]])]]
        }
        Zdm <- demeanlist(as.matrix(Z), list(fe.newX[fe.newX %ni% obj$fe$fe_var]))
        B <- foreach(i = 1:length(nlayers), .combine = c) %do% {obj$parlist[[i]]$beta}
        ydm_test <- demeanlist(y_test[fe.newX %ni% obj$fe$fe_var], list(fe.newX[fe.newX %ni% obj$fe$fe_var]))
        fe <- (y_test[fe.newX %ni% obj$fe$fe_var]-ydm_test) - 
          MatMult(as.matrix(Z)-Zdm, as.matrix(c(obj$parlist$beta_param, B)))
        FEs_to_append <- summaryBy(fe~fe_var, keep.names = T,
                                   data = data.frame(fe = fe, fe_var = fe.newX[fe.newX %ni% obj$fe$fe_var]))
        FEs_to_merge <- rbind(FEs_to_merge, FEs_to_append)
      }
    } else {FEs_to_merge <- NULL}
    #(predfun is defined below)
    yhat <- predfun_multinet(plist = plist, obj = obj, newX = newX, fe.newX = fe.newX
                    , new.param = new.param, FEs_to_merge = FEs_to_merge) 
    if (se.fit == FALSE){
      return(yhat)
    } else {
      if (is.null(obj$vcs)){
        stop("No vcov matrices in object.  Can't calculate se's")
      }
      #predicted pseudovariables
      if (any(grepl('Jac', names(obj$vcs)))){#only calculate the jacobian of the new obs if you have to
        if (numerical_jacobian == FALSE){
          J <- Jacobian.panelNNET(obj, numerical = FALSE, parallel = parallel_jacobian
            , step = 1e-9, newX = newX, new.param = new.param, fe.newX = fe.newX)
        } else {
          J <- jacobian(predfun, pvec, obj = obj, newX = newX, fe.newX = fe.newX
            , new.param = new.param, FEs_to_merge = FEs_to_merge)
          J <- J[,c(#re-order jacobian so that parametric terms are on the front, followed by top layer.
              which(grepl('param', names(pvec)))
            , which(grepl('beta', names(pvec)) & !grepl('param', names(pvec)))
            , which(!grepl('beta', names(pvec)))#no particular order to lower-level parameters
          )]
        }
      }
      #predicted top-level variables
      X <- as.matrix(predfun(pvec, obj = obj, newX = newX, fe.newX = fe.newX
          , new.param = new.param, FEs_to_merge = FEs_to_merge, return_toplayer = TRUE))
      vcnames <- c()
      semat <- foreach(i = 1:length(obj$vcs), .combine = cbind, .errorhandling = 'remove') %do% {
        if (grepl('OLS', names(obj$vcs)[i])){
          se <- foreach(j = 1:nrow(X), .combine = c)%do% {
            sqrt(X[j,, drop = FALSE] %*% obj$vcs[[i]] %*% t(X[j,, drop = FALSE]))
          }
        } else {
          se <- foreach(j = 1:nrow(J), .combine = c)%do% {
            sqrt(J[j,, drop = FALSE] %*% obj$vcs[[i]] %*% t(J[j,, drop = FALSE]))
          }
        }
        vcnames[i] <- names(obj$vcs)[i]
        return(se)
      }
      if (any(is.na(vcnames))){warning("One or more VCV has negative diagonals")}
      colnames(semat) <- vcnames[!is.na(vcnames)]
    }
    return(cbind(yhat, semat))
  }
}

#prediction function, potentially for the Jacobian
predfun_multinet <- function(plist, obj, newX = NULL, fe.newX = NULL, new.param = NULL,
                    FEs_to_merge = NULL, return_toplayer = FALSE, convolutional = NULL){
  if (obj$activation == 'tanh'){
    activ <- tanh
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
  }
  if (obj$activation == 'relu'){
    activ <- relu
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
  }
  nlayers <- foreach(i = 1:(length(obj$parlist)-1), .combine = c) %do% {length(obj$hidden_layers[[i]])}
  # rescale new data to scale of training data
  D <- foreach(i = 1:length(obj$X)) %do% {
    sweep(sweep(newX[[i]], 2, STATS = attr(obj$X[[i]], "scaled:center"), FUN = '-'), 2, STATS = attr(obj$X[[i]], "scaled:scale"), FUN = '/')
  }
  if (!is.null(obj$param)){
    P <- sweep(sweep(new.param, 
                     MARGIN = 2, 
                     STATS = attr(obj$param, "scaled:center"), 
                     FUN = '-'), 
               MARGIN = 2, 
               STATS = attr(obj$param, "scaled:scale"), 
               FUN = '/')
  } else {P <- NULL}
  if (is.null(FEs_to_merge)){# add intercept when needed
    P <- matrix(rep(1, nrow(D[[1]])))
  }
  # compute hidden layers
  HL <- calc_hlayers(parlist = obj$parlist, 
                     X = D, 
                     param = P, 
                     fe_var = obj$fe_var, 
                     nlayers = nlayers,
                     activation = obj$activation)
  D <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
    HL[[i]][[length(HL[[i]])]]
  }
  if (return_toplayer == TRUE){
    return(D)
  }
  B <- foreach(i = 1:length(nlayers), .combine = c) %do% {obj$parlist[[i]]$beta}
  B <- c(obj$parlist$beta_param, B)
  xpart <- MatMult(cbind(P, D), B)
  if (is.null(obj$fe)){
    yhat <- xpart
  } else {
    nd <- data.frame(fe.newX, xpart = as.numeric(xpart), id = 1:length(fe.newX))
    nd <- merge(nd, FEs_to_merge, by.x = 'fe.newX', by.y = 'fe_var', all.x = TRUE, sort = FALSE)
    nd <- nd[order(nd$id),]
    yhat <- nd$fe + nd$xpart
  }
  #otherwise...
  return(yhat)
}


#prediction function, potentially for the Jacobian
predfun <- function(plist, obj, newX = NULL, fe.newX = NULL, new.param = NULL,
                    FEs_to_merge = NULL, return_toplayer = FALSE, convolutional = NULL){
  if (obj$activation == 'tanh'){
    activ <- tanh
  }
  if (obj$activation == 'logistic'){
    activ <- logistic
  }
  if (obj$activation == 'relu'){
    activ <- relu
  }
  if (obj$activation == 'lrelu'){
    activ <- lrelu
  }
  # rescale new data to scale of training data
  D <- sweep(sweep(newX, 2, STATS = attr(obj$X, "scaled:center"), FUN = '-'), 2, STATS = attr(obj$X, "scaled:scale"), FUN = '/')
  if (!is.null(obj$param)){
    P <- sweep(sweep(new.param, 
                     MARGIN = 2, 
                     STATS = attr(obj$param, "scaled:center"), 
                     FUN = '-'), 
               MARGIN = 2, 
               STATS = attr(obj$param, "scaled:scale"), 
               FUN = '/')
  } else {P <- NULL}
  # compute hidden layers
  HL <- calc_hlayers(parlist = obj$parlist, 
                    X = D, 
                    param = P, 
                    fe_var = obj$fe_var, 
                    nlayers = length(obj$hidden_layers)-!is.null(obj$convolutional),# subtract off 1 when convolutional because "nlayers" doesn't include conv layer
                    convolutional = obj$convolutional,
                    activation = obj$activation)
  D <- HL[[length(HL)]]

  if (return_toplayer == TRUE){
    return(D)
  }
  xpart <- MatMult(D, as.matrix(c(plist$beta_param, plist$beta)))
  if (is.null(obj$fe)){
    yhat <- xpart
  } else {
    nd <- data.frame(fe.newX, xpart = as.numeric(xpart), id = 1:length(fe.newX))       
    nd <- merge(nd, FEs_to_merge, by.x = 'fe.newX', by.y = 'fe_var', all.x = TRUE, sort = FALSE)
    nd <- nd[order(nd$id),]
    yhat <- nd$fe + nd$xpart
  }
  #otherwise...
  return(yhat)
}



















