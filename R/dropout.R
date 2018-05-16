
# this file contains helper functions for managing drouput

# generate a droplist
gen_droplist <- function(dropout_hidden, dropout_input, nlayers, hlayers, X){
  if (dropout_hidden < 1){
    Xd <- dropinp <- list()
    DL <- foreach(i = 1:length(nlayers)) %do% {
      droplist <- lapply(hlayers[[i]], function(x){
        todrop <- as.logical(rbinom(ncol(x), 1, dropout_hidden))
        if (all(todrop==FALSE)){#ensure that at least one unit is present
          todrop[sample(1:length(todrop))] <- TRUE
        }
        return(todrop)
      })
      todrop <- rbinom(ncol(X[[i]]), 1, dropout_input)
      if (all(todrop==FALSE)){# ensure that at least one unit is present
        todrop[sample(1:length(todrop))] <- TRUE
      }
      droplist <- c(list(as.logical(todrop)), droplist)
      return(droplist)
    }
    droplist <- DL
  } else {droplist <- NULL}
  return(droplist)
}

# subset parlist
# this function won't work for non-multinets, until multinet is generalized to work for lists of length 1
drop_parlist <- function(parlist, droplist, nlayers){
  if (!is.null(droplist)){
    for (j in 1:length(nlayers)){
      if (nlayers[[j]] > 1){
        #drop from parameter list emanating from input
        parlist[[j]][[1]] <- parlist[[j]][[1]][c(TRUE,droplist[[j]][[1]]),droplist[[j]][[2]]]
        # drop from subsequent parameter matrices
        for (i in 2:(nlayers[[j]])){
          parlist[[j]][[i]] <- parlist[[j]][[i]][c(TRUE, droplist[[j]][[i]]), droplist[[j]][[i+1]], drop = FALSE]
        }
      } else { # one layer
        parlist[[j]][[1]] <- parlist[[j]][[1]][c(TRUE,droplist[[j]][[1]]),
                                           droplist[[j]][[nlayers]], 
                                           drop = FALSE]
      }
      parlist[[j]]$beta <- parlist[[j]]$beta[droplist[[j]][[nlayers[[j]]+1]]]
    }
  }
  return(parlist)
}

# subset hidden layers and data
drop_data <- function(hlayers, droplist, X){
  foreach(i = 1:length(droplist)) %do% {
    foreach(j = 1:length(droplist[[i]])) %do% {
      if (j == 1){Z <- X[[i]]} else {Z <- hlayers[[i]][[j-1]]}
      Z[,droplist[[i]][[j]], drop = FALSE]
    }
  }
}

# reconstitute full gradient after doing dropout
reconstitute <- function(dropped, droplist, full_old_parlist, nlayers){
# dropped <- grads_p
# full_old_parlist <- parlist
  if (!is.null(droplist)){
    # if (!is.null(full_old_parlist$beta_param)){
    #   BP <- full_old_parlist$beta_param      
    # }
    emptygrads <- recursive_mult(full_old_parlist, 0)
    for (j in 1:length(nlayers)){
      if (nlayers[[j]] > 1){
        emptygrads[[j]][[1]][c(TRUE,droplist[[j]][[1]]),droplist[[j]][[2]]] <- dropped[[j]][[1]]
        for (i in 2:(nlayers[[j]])){
          emptygrads[[j]][[i]][c(TRUE, droplist[[j]][[i]]), droplist[[j]][[i+1]]] <- dropped[[j]][[i]]
        }
      } else { #for one-layer networks
        emptygrads[[j]][[1]][c(TRUE,dropinp[[j]]),
                             droplist[[j]][[1]]] <- dropped[[j]][[1]]
      }
      #top-level
      emptygrads[[j]]$beta <- NULL
      emptygrads[[j]][[nlayers[[j]] + 1]] <- matrix(rep(0, length(full_old_parlist[[j]]$beta))) #empty
      emptygrads[[j]][[nlayers[[j]] + 1]][droplist[[j]][[nlayers[[j]]+1]]] <- dropped[[j]][[nlayers[[j]] + 1]]
    }
    if (length(emptygrads)>length(nlayers)){ # this will happen when there is a beta_param term
      emptygrads$beta_param <- dropped[[length(dropped)]]
    }
    return(emptygrads)
  } else {return(dropped)} 
}




