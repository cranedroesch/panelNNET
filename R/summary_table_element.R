summary_table_element <-
function(vc, parm){#Function used internally to the infernce == TRUE argument
  if (inherits(vc, 'error') | any(diag(vc)<=0)){
    'Error!  Probably an ill-conditioned covariance matrix'
  } else {
    se <- sqrt(diag(vc))[1:length(parm)]#The first vc elements are always the parametric terms, followed by any treatment dummy
    p <- 2*pnorm(-abs(parm/se))
    stars <- rep('',length(parm))
    stars[p<.1] <- '*'
    stars[p<.05] <- '**'
    stars[p<.01] <- '***'
    list(signif(se, 4), signif(p,4), stars)
  }
}


