calc_grads<- function(plist, hlay, Xd, y, yhat, droplist = NULL, nlayers, convolutional = NULL, activ_prime, normalize = TRUE){
# plist <- parlist
# hlay <- hlayers
# Xd <- X
# droplist <- NULL
# curBat = NULL
# hlay = hlay
# yhat = yhat[curBat]
# curBat = curBat
  # if (is.null(yhat)){yhat <- getYhat(plist, hlay = hlay)}
  NL <- nlayers + as.numeric(!is.null(convolutional))
  grads <- foreach(p = 1:length(NL)) %do% {
    grads <- grad_stubs <- vector('list', NL[p] + 1)
    grad_stubs[[length(grad_stubs)]] <- getDelta(as.matrix(y), yhat)
    for (i in NL[p]:1){
      if (i == NL[p]){outer_param = as.matrix(c(plist[[p]]$beta))} else {outer_param = plist[[p]][[i+1]]}
      if (i == 1){lay = Xd[[p]]} else {lay= hlay[[p]][[i-1]]}
      #add the bias
      lay <- cbind(1, lay) #add bias to the hidden layer
      if (i != NL[p]){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
      grad_stubs[[i]] <- activ_prime(MatMult(lay, plist[[p]][[i]])) * MatMult(grad_stubs[[i+1]], Matrix::t(outer_param))
    }
    grad_stubs <- lapply(grad_stubs, as.matrix)
    for (i in 1:length(grad_stubs)){
      if (i == 1){lay = as.matrix(Xd[[p]])} else {lay= hlay[[p]][[i-1]]}
      if (i != length(grad_stubs)){# don't add bias term to top layer when there are fixed effects present
        lay <- cbind(1, lay) #add bias to the hidden layer
      }
      grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
    }
    return(grads)
  }
  # add on parametric gradients
  if (!is.null(hlay$param)){
    grads[[length(grads)+1]] <- MatMult(t(hlay$param), getDelta(as.matrix(y), yhat)) 
  }
  if (normalize == TRUE){
    fac <- 1/mean(unlist(grads))
    grads <- recursive_mult(grads, fac)
  }
  return(grads)
}


# old version:

#   calc_grads<- function(plist, hlay, Xd, y, yhat, curBat = NULL, droplist = NULL, nlayers){
# # plist <- parlist
# # hlay <- hlayers
# # Xd <- X
# # droplist <- NULL
# # curBat = NULL
# # hlay = hlay
# # yhat = yhat[curBat]
# # curBat = curBat
#     # #subset the parameters and hidden layers based on the droplist
#     # if (!is.null(droplist)){
#     #   if (length(nlayers)>1){
#     #     for (j in 1:length(nlayers)){
#     #       if (nlayers[[j]] > 1){
#     #         #drop from parameter list emanating from input
#     #         plist[[j]][[1]] <- plist[[j]][[1]][c(TRUE,dropinp[[j]]),droplist[[j]][[1]]]
#     #         # drop from subsequent parameter matrices
#     #         if (nlayers[[j]]>2){
#     #           for (i in 2:(nlayers[[j]]-1)){
#     #             plist[[j]][[i]] <- plist[[j]][[i]][c(TRUE, droplist[[j]][[i-1]]), droplist[[j]][[i]], drop = FALSE]
#     #           }
#     #         }
#     #       } else { # one layer
#     #         plist[[j]][[1]] <- plist[[j]][[1]][c(TRUE,dropinp[[j]]),
#     #                                            droplist[[j]][[nlayers]], 
#     #                                            drop = FALSE]
#     #       }
#     #       # special treatment for the top layer
#     #       plist[[j]][[nlayers[[j]]]] <- plist[[j]][[nlayers[[j]]]][c(TRUE, droplist[[j]][[nlayers[[j]]-1]]), 
#     #                                                                droplist[[j]][[nlayers[[j]]]], 
#     #                                                                drop = FALSE]
#     #       plist[[j]]$beta <- plist[[j]]$beta[droplist[[j]][[nlayers[[j]]]]]
#     #     }
#     #   } else {
#     #     Xd <- X[,dropinp, drop = FALSE]
#     #     if (nlayers > 1){
#     #       #drop from parameter list emanating from input
#     #       plist[[1]] <- plist[[1]][c(TRUE,dropinp),droplist[[1]]]
#     #       # drop from subsequent parameter matrices
#     #       if (nlayers>2){
#     #         for (i in 2:(nlayers-1)){
#     #           plist[[i]] <- plist[[i]][c(TRUE, droplist[[i-1]]), droplist[[i]], drop = FALSE]
#     #         }
#     #       }
#     #       plist[[nlayers]] <- plist[[nlayers]][c(TRUE, droplist[[nlayers-1]]), 
#     #                                            droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])], 
#     #                                            drop = FALSE]
#     #     } else { #for one-layer networks
#     #       #drop from parameter list emanating from input
#     #       plist[[1]] <- plist[[1]][c(TRUE,dropinp),
#     #                                droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])], 
#     #                                drop = FALSE]
#     #     }
#     #     # manage parametric/nonparametric distinction in the top layer
#     #     plist$beta <- plist$beta[droplist[[nlayers]][(length(parlist$beta_param)+1):length(droplist[[nlayers]])]] 
#     #   }
#     # } else {Xd <- X}#for use below...  X should be safe given scope, but extra assignment is cheap here
#     if (!is.null(curBat)){CB <- function(x){x[curBat,,drop = FALSE]}} else {CB <- function(x){x}}
#     # if (is.null(yhat)){yhat <- getYhat(plist, hlay = hlay)}
#     NL <- nlayers + as.numeric(!is.null(convolutional))
#     if (length(NL)>1){
#       grads <- foreach(p = 1:length(NL)) %do% {
#         grads <- grad_stubs <- vector('list', NL[p] + 1)
#         grad_stubs[[length(grad_stubs)]] <- getDelta(CB(as.matrix(y)), yhat)
#         for (i in NL[p]:1){
#           if (i == NL[p]){outer_param = as.matrix(c(plist[[p]]$beta))} else {outer_param = plist[[p]][[i+1]]}
#           if (i == 1){lay = CB(Xd[[p]])} else {lay= CB(hlay[[p]][[i-1]])}
#           #add the bias
#           lay <- cbind(1, lay) #add bias to the hidden layer
#           if (i != NL[p]){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
#           grad_stubs[[i]] <- activ_prime(MatMult(lay, plist[[p]][[i]])) * MatMult(grad_stubs[[i+1]], Matrix::t(outer_param))
#         }
#         grad_stubs <- lapply(grad_stubs, as.matrix)
#         for (i in 1:length(grad_stubs)){
#           if (i == 1){lay = as.matrix(CB(Xd[[p]]))} else {lay= CB(hlay[[p]][[i-1]])}
#           if (i != length(grad_stubs)){# don't add bias term to top layer when there are fixed effects present
#             lay <- cbind(1, lay) #add bias to the hidden layer
#           }
#           grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
#         }
#         return(grads)
#       }
#       # add on parametric gradients
#       grads[[length(grads)+1]] <- MatMult(t(CB(hlay$param)), getDelta(CB(as.matrix(y)), yhat)) 
#     } else {
#       grads <- grad_stubs <- vector('list', NL + 1)
#       grad_stubs[[length(grad_stubs)]] <- getDelta(CB(as.matrix(y)), yhat)
#       for (i in NL:1){
#         # print(i)
#         if (i == NL){outer_param = as.matrix(c(plist$beta))} else {outer_param = plist[[i+1]]}
#         if (i == 1){lay = CB(Xd)} else {lay= CB(hlay[[i-1]])}
#         #add the bias
#         lay <- cbind(1, lay) #add bias to the hidden layer
#         if (i != NL){outer_param <- outer_param[-1,, drop = FALSE]}      #remove parameter on upper-layer bias term
#         grad_stubs[[i]] <- activ_prime(MatMult(lay, plist[[i]])) * MatMult(grad_stubs[[i+1]], Matrix::t(outer_param))
#       }
#       # multiply the gradient stubs by their respective layers to get the actual gradients
#       # first coerce them to regular matrix classes so that the C code for matrix multiplication can speed things up
#       grad_stubs <- lapply(grad_stubs, as.matrix)
#       hlay <- lapply(hlay, as.matrix)
#       for (i in 1:length(grad_stubs)){
#         if (i == 1){lay = as.matrix(CB(Xd))} else {lay= CB(hlay[[i-1]])}
#         if (i != length(grad_stubs)){# don't add bias term to top layer when there are fixed effects present
#           lay <- cbind(1, lay) #add bias to the hidden layer
#         }
#         grads[[i]] <- eigenMapMatMult(t(lay), as.matrix(grad_stubs[[i]]))
#       } 
#     }
#     # #process the gradients for the convolutional layers
#     # if (!is.null(convolutional)){
#     #   if (!is.null(droplist)){
#     #     warning("dropout not yet made to work with conv nets")
#     #   }
#     #   #mask out the areas not in use
#     #   gg <- grads[[1]] * convMask
#     #   #gradients for conv layer.  pooling via rowMeans
#     #   grads_convParms <- foreach(i = 1:convolutional$Nconv) %do% {
#     #     idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
#     #     rowMeans(foreach(j = idx, .combine = cbind) %do% {x <- gg[,j]; x[1] <- -999; x[x!=0][-1]})
#     #   }
#     #   grads_convBias <- foreach(i = 1:convolutional$Nconv, .combine = c) %do% {
#     #     idx <- (1+N_TV_layers*(i-1)):(N_TV_layers*i)
#     #     mean(gg[1,idx])
#     #   }
#     #   # make the layer
#     #   convGrad <- makeConvLayer(grads_convParms, grads_convBias)
#     #   #set the gradients on the time-invariant terms to zero
#     #   convGrad[,(N_TV_layers * convolutional$Nconv+1):ncol(convGrad)] <- 0
#     #   grads[[1]] <- convGrad
#     # }
#     return(grads)
#   }