
vcov.panelNNET <-
function(obj, option, J = NULL, edf_J = NULL, edf_X = NULL){
#obj <- pnn
#option = 'cluster'
  e <- obj$y - obj$yhat
  if (grepl('Jacobian', option)){
    if (is.null(J)){stop('must supply Jacobian if requesting a Jacobian approx')}
    #put together penalty factor
    D <- rep(obj$lam, ncol(J))
    if (is.null(obj$fe_var)){
      pp <- c(0, obj$parapen) #never penalize the intercept
    } else {
      pp <- obj$parapen #parapen
    }
    D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
    #use the EDF corresponding to the Jacobian approximation
    edf <- obj$edf_J
    #compute "bread"
    bread <- solve(Matrix::t(J) %*% J + diag(D))
    if (option == 'Jacobian_homoskedastic'){
      vcov <- sum(e^2)/(length(e) - edf) * bread
    }
    if (option == 'Jacobian_sandwich'){
      meat <- foreach(i = 1:length(e), .combine = '+') %do% {
        e[i]^2*J[i,] %*% Matrix::t(J[i,])
      }
      vcov <- (length(e)-1)/(length(e) - edf) * bread %*% meat %*% bread
    }
    if (option == 'Jacobian_cluster'){
      G <- length(unique(obj$fe_var))
      meat <- foreach(i = 1:G, .combine = '+')%do%{
        ei <- e[obj$fe_var == unique(obj$fe_var)[i]]
        Ji <- J[obj$fe_var == unique(obj$fe_var)[i],,drop = FALSE]
        
        t(as.matrix(Ji)) %*% as.matrix(ei) %*% t(as.matrix(ei)) %*% as.matrix(Ji)
      }
      vcov <- G/(G-1)*(length(e) - 1)/(length(e) - edf) * bread %*% meat %*% bread
    }
  #if using one of the OLS approximations
  } else {
    if (is.null(obj$edf_X)){stop('need to put edf_X in object, probably by runninng "do_inference" function')}
    J <- obj$hidden_layers[[length(obj$hidden_layers)]]
    #put together penalty factor
    D <- rep(obj$lam_X, ncol(J))
    if (is.null(obj$fe_var)){
      pp <- c(0, obj$parapen) #never penalize the intercept
    } else {
      pp <- obj$parapen #parapen
    }
    D[1:length(pp)] <- D[1:length(pp)]*pp #incorporate parapen into diagonal of covmat
    #use the EDF corresponding to the Ridge approximation
    edf <- obj$edf_X
    #compute "bread"
    bread <- solve(Matrix::t(J) %*% J + diag(D))
    if (option == 'OLS'){
      vcov <- sum(e^2)/(length(e) - edf) * bread
    }
    if (option == 'sandwich'){
      meat <- foreach(i = 1:length(e), .combine = '+') %do% {
        e[i]^2*J[i,] %*% Matrix::t(J[i,])
      }
      vcov <- (length(e)-1)/(length(e) - edf) * bread %*% meat %*% bread
    }
    if (option == 'cluster'){
      G <- length(unique(obj$fe_var))
      meat <- foreach(i = 1:G, .combine = '+')%do%{
        ei <- e[obj$fe_var == unique(obj$fe_var)[i]]
        Ji <- J[obj$fe_var == unique(obj$fe_var)[i],,drop = FALSE]
        t(Ji) %*% ei %*% t(ei) %*% Ji
      }
      vcov <- G/(G-1)*(length(e) - 1)/(length(e) - edf) * bread %*% meat %*% bread
    }
  }
  return(vcov)
}





