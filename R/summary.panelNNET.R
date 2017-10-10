summary.panelNNET <-
function(x, ...){
  if(is.null(x$vcs)){
    infstrings <- NULL
  } else {
    dparm <- parm <- c(x$parlist$beta_param)
    #scaling
    scalefac <- c(rep(attr(x$param, "scaled:scale"), ncol(x$param)+is.null(x$fe_var)))
    dparm <- dparm/scalefac
    #Interence strings -- to send to `writelines`
    infstrings <- "\nParametric Estimates:\n"  
    #Parameter names and variance estimate labels...
    if (is.null(colnames(x$param))){
      if (is.null(x$fe_var)){
        #labs <- c('LTE, homoskedastic vcv', 'LTE, sandwich vcv', 'OLS/ridge, homoskedastic vcv', 'OLS/ridge, sandwich vcv')
        labs <- names(x$vcs)
        parnames <- c('(Intercept)', paste0('V', 1:ncol(x$param)))
      } else {
        #labs <- c('LTE, homoskedastic vcv', 'LTE, sandwich vcv', 'LTE, cluster vcv', 'OLS/ridge, homoskedastic vcv', 'OLS/ridge, sandwich vcv', 'OLS/ridge, cluster vcv')
        labs <- names(x$vcs)
        parnames <- paste0('V', 1:ncol(x$param))
      }
    } else {
      labs <- names(x$vcs)
      parnames <- colnames(x$param)    
    }

    for (i in 1:length(labs)){
      s <- summary_table_element(x$vcs[[i]], parm)
      infstrings <- paste0(infstrings, "-----------------------------------------------------------\n")
      infstrings <- paste0(infstrings, labs[i], "\n")
      infstrings <- paste0(infstrings,  paste(rep(' ',max(nchar(parnames))), collapse = ""), "\t\t\tEst\t\tSE\t\tpval\t\t", "\n")
      #scale factor for parameters
      for (j in 1:length(s[[1]])){
        if (length(s) == 1){
          infstrings <- paste0(infstrings, "\t", s, "\n")
        } else {
          if (j==1){
            s[[1]] <- s[[1]]/scalefac
          }
          #futz with notation
          pnum <- dosci(signif(dparm[j],3),3)
          infstrings <- paste0(infstrings
            ,  "\t", parnames[j], "\t\t"
            , pnum, paste(rep(' ',10-nchar(pnum)), collapse = ""), "\t"
            , dosci(signif(s[[1]][j],3),3), paste(rep(' ',10-nchar(dosci(signif(s[[1]][j],3),3))), collapse = ""), "\t"
            , dosci(signif(s[[2]][j],3),3), "\t"
            , s[[3]][j], "\t"
          )
          infstrings <- paste0(infstrings, "\n")
        }
      }
    }
  }
  with(x, writeLines(paste0(
      "*******************************************\n"
    , "Panel data neural network \n"
    , 'Lambda = ',lam, "\n"
    , "Hidden units: ",paste(hidden_units, collapse = ' '), "\n"
    , "Converged ", as.logical(converged), "\n"
    , "mse is ",mse, "\n"
    , "loss is ",loss, "\n"
    , "Number of fixed effects: ", length(unique(fe_var)), "\n"
    , "Number of linear terms: ", ncol(param), "\n"
    , "Number of terms in the base layer: ", ncol(X), "\n"
    , "Gradient descent method: ", ifelse(RMSprop, "RMSprop", "Batch gradient descent"), "\n"
    , "Convergence tolerance: ", convtol, "\n"
    , infstrings
    , "*******************************************\n"
  )))
}


