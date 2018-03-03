



predict.panelNNET <-
function(obj, newX = NULL, fe.newX = NULL, new.param = NULL, se.fit = FALSE
         , numerical_jacobian = FALSE, parallel_jacobian = FALSE, convolutional = NULL){
# obj = pnn
# newX = Z[v,]
# new.param = P[v,]
# fe.newX = id[v]
# se.fit = T
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
    if (!all(unique(fe.newX) %in% unique(obj$fe$fe_var))){
      stop('New data has cross-sectional units not observed in training data')
    }
    #Scale the new data by the scaling rules in the training data
    plist <- as.relistable(obj$parlist) # pull out from list 
    pvec <- unlist(plist)
    #prepare fe's in advance...
    if (!is.null(obj$fe)){
      FEs_to_merge <- foreach(i = 1:length(unique(obj$fe$fe_var)), .combine = rbind) %do% {
        #Because of numerical error, fixed effects within units can sometimes be slightly different.  This averages them.
        data.frame(unique(obj$fe$fe_var)[i], mean(obj$fe$fe[obj$fe$fe_var == unique(obj$fe$fe_var)[i]]))
      }
      colnames(FEs_to_merge) <- c('fe_var','fe')
    } else {FEs_to_merge <- NULL}
    #(predfun is defined below)
    yhat <- predfun(plist = plist, obj = obj, newX = newX, fe.newX = fe.newX
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
  }
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



















