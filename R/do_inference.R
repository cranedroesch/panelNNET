


do_inference <- function(obj, numerical = FALSE, parallel = TRUE
  , step = 1e-9, J = NULL, verbose = FALSE, OLS_only = FALSE){
  #Compute Jacobian if not supplied and asked-for
  if (is.null(J) & OLS_only == FALSE){
    if (verbose){print("computing Jacobian")}
    J <- Jacobian.panelNNET(obj, parallel, step, numerical)
  }
  #top layer for OLS approximation
  X <- obj$hidden_layers[[length(obj$hidden_layers)]]
  #residuals, for sigma
  res <- with(obj, y - yhat)
  #calculate EDF and add to output
  if (OLS_only == FALSE){
    obj$J <- J #save the jacobian in the object
    #de-mean, if fixed effects
    if (is.null(obj$fe_var)){
      Jdm <- J
    } else {
      Jdm <- demeanlist(J, list(obj$fe_var))
    }
    #do SVD
    svJ <- svd(Jdm)
    #put together diagonal of penalty matrix
    D <- rep(obj$lam, ncol(J))
    if (is.null(obj$fe_var)){
      pp <- c(0, obj$parapen) #never penalize the intercept
    } else {
      pp <- obj$parapen #parapen
    }
    D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
    obj$edf_J <- sum(svJ$d^2/(svJ$d^2+D))
    obj$sigma2_J <- sum(res^2)/(nrow(X) - obj$edf_J)
  }
  #de-mean, if fixed effects
  if (is.null(obj$fe_var)){
    Xdm <- X
    targ <- obj$y
  } else {
    Xdm <- demeanlist(as.matrix(X), list(obj$fe_var))
    targ <- demeanlist(obj$y, list(obj$fe_var))
  }
  #get implicit lambda for top level ridge regression
  if (verbose){print("getting implicit lambda")}
  D <- rep(1, ncol(Xdm))
  if (is.null(obj$fe_var)){
    pp <- c(0, obj$parapen) #never penalize the intercept
  } else {
    pp <- obj$parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  if (obj$lam > 0) {
    constraint <- sum(c(obj$parlist$beta_param*obj$parapen, obj$parlist$beta)^2)
    #function to find implicit lambda
    f <- function(lam){
      bi <- solve(crossprod(Xdm) + diag(D)*lam) %*% t(Xdm) %*% targ
      (t(bi*D) %*% (bi*D) - constraint)^2
    }
    #optimize it
    o <- optim(par = obj$lam, f = f, method = 'Brent', lower = obj$lam, upper = 1e9)
    obj$lam_X <- o$par
  } else {
    obj$lam_X <- 0
  }
  #do SVD
  if (verbose){print('starting svd')}
  svX <- svd(Xdm)
  D <- rep(obj$lam_X, ncol(X))
  if (is.null(obj$fe_var)){
    pp <- c(0, obj$parapen) #never penalize the intercept
  } else {
    pp <- obj$parapen #parapen
  }
  D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
  obj$edf_X <- sum(svX$d^2/(svX$d^2+D))
  obj$sigma2_X <- sum(res^2)/(nrow(X) - obj$edf_X)
  #empty list of vcovs
  vcs <- list()
  if (verbose){print("getting vcovs")}
  #calculate Jacobian-based vcovs
  if (OLS_only == FALSE){
    vcs[["vc.JacHomo"]] <- tryCatch(vcov.panelNNET(obj, 'Jacobian_homoskedastic', J = J), error = function(e)e, finally = NULL)
    vcs[["vc.JacSand"]] <- tryCatch(vcov.panelNNET(obj, 'Jacobian_sandwich', J = J), error = function(e)e, finally = NULL)    
    if (!is.null(obj$fe_var)){
      vcs[["vc.JacClus"]] <- tryCatch(vcov.panelNNET(obj, 'Jacobian_cluster', J = J), error = function(e)e, finally = NULL)
    }
  } 
  #calculate OLS aproximations
  vcs[["vc.OLSHomo"]] <- tryCatch(vcov.panelNNET(obj, 'OLS'), error = function(e)e, finally = NULL)
  vcs[["vc.OLSSand"]] <- tryCatch(vcov.panelNNET(obj, 'sandwich'), error = function(e)e, finally = NULL)
  if (!is.null(obj$fe_var)){
    vcs[["vc.OLSClus"]] <- tryCatch(vcov.panelNNET(obj, 'cluster'), error = function(e)e, finally = NULL)
  }
  obj$vcs <- vcs
  return(obj)
}



