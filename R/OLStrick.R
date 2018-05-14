
OLStrick_function <- function(parlist, hidden_layers, y, fe_var, lam, parapen, penalize_toplayer, nlayers){
  # parlist <- pnn$parlist
  # hidden_layers <- pnn$hidden_layers
  # y = pnn$y
  # fe_var <- pnn$fe_var
  # lam <- pnn$lam
  # parapen <- pnn$parapen
  # hidden_layers <- hlayers
  const <- parlist$beta_param*parapen
  if (penalize_toplayer == TRUE){
    for (i in 1:(length(parlist)-1)){
      const <- c(const, parlist[[i]]$beta)
    }
  }
  constraint <- sum(const^2)
  Zdm <- foreach(i = 1:length(nlayers), .combine = cbind) %do% {
    hidden_layers[[i]][[length(hidden_layers[[i]])]]
  }
  Zdm <- cbind(hidden_layers$param, Zdm)
  if (!is.null(fe_var)){
    Zdm <- demeanlist(Zdm, list(fe_var))
    targ <- demeanlist(y, list(fe_var))      
  }  else {
    targ <- y
  }   
  #set up the penalty vector
  D <- rep(1, ncol(Zdm))
  D[1:length(parapen)] <- D[1:length(parapen)]*parapen #incorporate parapen into diagonal of covmat
  if (penalize_toplayer == FALSE){
    D <- D*0
  }
  if (is.null(fe_var)){
    D[1] <- 0
  }
  # find implicit lambda
  b <- c(unlist(parlist)[grepl("beta", names(unlist(parlist)))])
  b <- c(b[grepl("param", names(b))], b[!grepl("param", names(b))])
  # b <- c(parlist$beta_param, parlist$beta)
  Zty <- MatMult(t(Zdm), targ)
  ZtZ <- MatMult(t(Zdm), Zdm)
  if (constraint >0){
    f <- function(lam){
      bi <- tryCatch(as.numeric(MatMult(solve(ZtZ + diag(D)*as.numeric(lam)), Zty)), error = function(e){b})
      (crossprod(bi*D) - constraint)^2
    }
    o <- optim(par = lam, f = f, method = 'Brent', lower = lam, upper = 1e9)
    newlam2 <- o$par
    #New top-level params
    b <- tryCatch(as.numeric(MatMult(solve(ZtZ + diag(D)*as.numeric(newlam2)), Zty)),
                  error = function(e){b})
  } else {
    b <- tryCatch(as.numeric(MatMult(solve(ZtZ), Zty)),
                  error = function(e){b})
  }
  if (inherits(b, "error")){
    print("singularity in OLStrick!")
    return(parlist)
    # b <- as.numeric(MatMult(ginv(ZtZ + diag(D)*as.numeric(newlam)), Zty))
  }    
  parlist$beta_param <- b[1:length(parlist$beta_param)]
  leftoff <- length(parlist$beta_param)
  for (i in 1:(length(parlist)-1)){
    idx <- (leftoff+1):(leftoff+length(parlist[[i]]$beta))
    parlist[[i]]$beta <- b[idx]
    leftoff <- max(idx)
  }
  return(parlist)
}

