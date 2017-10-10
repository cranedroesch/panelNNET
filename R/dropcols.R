dropcols <-
function(bx){#This is a recursive function...
  bx <- bx[,apply(bx,2,function(x){sum(is.nan(x))})==0] #remove NANs
	testcols <- function(ee) {#test columns for linear combs
		## split eigenvector matrix into a list, by columns
		evecs <- split(zapsmall(ee$vectors),col(ee$vectors))
		## for non-zero eigenvalues, list non-zero evec components
		mapply(function(val,vec) {
		if (val!=0) NULL else which(vec!=0)
			},zapsmall(ee$values),evecs)
	}
	m = crossprod(bx)
	ee <- eigen(m)
	lcs = unlist(testcols(ee))
	if (length(lcs)>0){
    todrop <- lcs[length(lcs)]
		bx = bx[,-todrop]
		dropcols(bx)
	} else {
		return(bx)
	}
}
