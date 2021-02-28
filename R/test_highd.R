rm(list = ls())
Rcpp::sourceCpp('src/smQuantile.cpp')

exam = function(beta, beta.hat) {
  TPR = sum(beta != 0 & beta.hat != 0) / sum(beta != 0)
  FPR = sum(beta == 0 & beta.hat != 0) / sum(beta == 0)
  err.itcp = abs(beta[1] - beta.hat[1])
  err.coef = norm(beta[-1] - beta.hat[-1], "2")
  return (list(TRP = TPR, FPR = FPR, err.itcp = err.itcp, err.coef = err.coef))
}

n = 200
d = 1000
s = 20
tau = 0.5
beta = c(rep(2, s + 1), rep(0, d - s))
beta0 = beta[1] + qt(tau, 2)

set.seed(2020)
X = matrix(rnorm(n * d), n, d)
Z = cbind(rep(1, n), X)
err = rt(n, 2)
Y = Z %*% beta + err 

#rst = smqrLasso(Z, Y, 0.038, tau, intercept = TRUE, itcpIncluded = TRUE)
#beta.smqr.lasso = as.numeric(rst$beta)

#rst = smqrSCAD(Z, Y, 0158, tau, intercept = TRUE, itcpIncluded = TRUE)
#beta.smqr.scad = as.numeric(rst$beta)

#rst = smqrMCP(Z, Y, 0.191, tau, intercept = TRUE, itcpIncluded = TRUE)
#beta.smqr.mcp = as.numeric(rst$beta)

## Cross-validation
report = matrix(0, 3, 4)

rst = cvSmqrLasso(Z, Y, tau = tau, intercept = TRUE, itcpIncluded = TRUE)
beta.smqr.lasso = as.numeric(rst$beta)
report[1, ] = unlist(exam(beta, beta.smqr.lasso))

rst = cvSmqrSCAD(Z, Y, tau = tau, intercept = TRUE, itcpIncluded = TRUE)
beta.smqr.scad = as.numeric(rst$beta)
report[2, ] = unlist(exam(beta, beta.smqr.scad))

rst = cvSmqrMCP(Z, Y, tau = tau, intercept = TRUE, itcpIncluded = TRUE)
beta.smqr.mcp = as.numeric(rst$beta)
report[3, ] = unlist(exam(beta, beta.smqr.mcp))

report = as.data.frame(report)
colnames(report) = c("TPR", "FPR", "err.itcp", "err.coef")
rownames(report) = c("Smqr-Lasso", "Smqr-SCAD", "Smqr-MCP")
report
