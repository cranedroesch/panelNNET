
getBLUP <- function(RE_sqrt = RE_sqrt, 
                    res = y-ydm, 
                    cs_var = cs_var, 
                    Z = Z){
  RE_vcv <- MatMult(t(RE_sqrt),RE_sqrt)
  Psi <- Diagonal(length(unique(cs_var))) %x% RE_vcv
  return(Psi %*% t(Z) %*% res / var(res))
}
