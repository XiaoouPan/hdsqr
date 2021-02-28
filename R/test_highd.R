rm(list = ls())
Rcpp::sourceCpp("src/hdsqr.cpp")

library(FHDQR)
library(MASS)
library(matrixStats)
library(glmnet)
library(caret)
library(rqPen)

exam = function(beta, beta.hat) {
  TPR = sum(beta != 0 & beta.hat != 0) / sum(beta != 0)
  FPR = sum(beta == 0 & beta.hat != 0) / sum(beta == 0)
  err.itcp = abs(beta[1] - beta.hat[1])
  err.coef = norm(beta[-1] - beta.hat[-1], "2")
  return (c(TPR, FPR, err.itcp, err.coef))
}

geneCov = function(p) {
  rst = matrix(0, p, p)
  for (i in 1:(p - 1)) {
    for (j in i:p) {
      rst[i, j] = rst[j, i] = 0.7^(abs(i - j))
    }
  } 
  return (rst)
}

n = 200
p = 500
s = 10
tau = 0.5
beta = c(rep(1.5, s + 1), rep(0, p - s))
h = (sqrt(s * log(p) / n) + (s * log(p) / n)^0.25) / 2
report = matrix(0, 5, 4)
kfolds = 5
M = 1

## Compare Lasso, SCAD, MCP
pb = txtProgressBar(style = 3)
for (m in 1:M) {
  set.seed(m)
  X = matrix(rnorm(n * p), n, p)
  Z = cbind(rep(1, n), X)
  df = 2
  err = rt(n, df) - qt(tau, df)
  Y = Z %*% beta + err 

  folds = createFolds(Y, kfolds, FALSE)
  
  fit = cv.glmnet(X, Y, nlambda = 50)
  lambdaSeq = fit$lambda
  beta.lasso = as.numeric(coef(fit, s = fit$lambda.min))
  report[1, ] = report[1, ] + exam(beta, beta.lasso)
  
  #fit = cv.qraenet(X, Y, lambda = lambdaSeq, tau = tau, nfolds = kfolds)
  #beta.qrLasso = as.numeric(coef(fit, s = fit$lambda.min))
  #report[2, ] = report[2, ] + exam(beta, beta.qrLasso)
  
  beta.sqLasso = cvSqrLasso(Z, Y, lambdaSeq, folds, tau, kfolds, h, phi0 = 0.01, gamma = 1.5)
  beta.sqLasso = as.numeric(beta.sqLasso)
  report[3, ] = report[3, ] + exam(beta, beta.sqLasso)
  
  beta.sqScad = cvSqrScad(Z, Y, lambdaSeq, folds, tau, kfolds, h, phi0 = 0.01, gamma = 1.5)
  beta.sqScad = as.numeric(beta.sqScad)
  report[4, ] = report[4, ] + exam(beta, beta.sqScad)
  
  beta.sqMcp = cvSqrMcp(Z, Y, lambdaSeq, folds, tau, kfolds, h, phi0 = 0.01, gamma = 1.5)
  beta.sqMcp = as.numeric(beta.sqMcp)
  report[5, ] = report[5, ] + exam(beta, beta.sqMcp)

  setTxtProgressBar(pb, m / M)  
}

report = as.data.frame(report / M)
colnames(report) = c("TPR", "FPR", "err.itcp", "err.coef")
rownames(report) = c("Lasso", "Qr-Lasso", "Sqr-Lasso", "Sqr-SCAD", "Sqr-MCP")
report
