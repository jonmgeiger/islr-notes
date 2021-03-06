Chapter 5 Labs
================

-   [The Validation Set Approach](#the-validation-set-approach)
-   [Leave-One-Out Cross-Validation](#leave-one-out-cross-validation)
-   [k-fold Cross-Validation](#k-fold-cross-validation)
-   [The Bootstrap](#the-bootstrap)

## The Validation Set Approach

We’ll be looking at the `Auto` data set to fit various linear models.
We’ll start out with setting the seed for reproducibility, using the
`sample()` function to get a random subset of all the examples. We’ll
fit a linear model predicting mpg from horsepower.

``` r
set.seed(1)
train <- sample(392, 196)

lm_fit <- lm(
    mpg ~ horsepower, data = Auto, subset = train)
```

We can now use `predict()` to estimate the response for all
observations, and calculate the MSE using the mean of squared residuals
for the test set:

``` r
mean((Auto$mpg - predict(lm_fit, Auto))[-train]^2)
```

    ## [1] 23.26601

If we instead fit quadratic or cubic fits, we get the following test
errors:

``` r
lm_fit2 <- lm(
    mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((Auto$mpg - predict (lm_fit2, Auto))[-train]^2)
```

    ## [1] 18.71646

``` r
lm_fit3 <- lm(
    mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((Auto$mpg - predict (lm_fit3, Auto))[-train]^2)
```

    ## [1] 18.71646

Choosing a different validation set gives, then, different error rates:

``` r
set.seed(2)
train <- sample(392, 196)

lm_fit <- lm(
    mpg ~ horsepower, data = Auto, subset = train)
mean((Auto$mpg - predict(lm_fit, Auto))[-train]^2)
```

    ## [1] 25.72651

``` r
lm_fit2 <- lm(
    mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((Auto$mpg - predict (lm_fit2, Auto))[-train]^2)
```

    ## [1] 20.43036

``` r
lm_fit3 <- lm(
    mpg ~ poly(horsepower, 2), data = Auto, subset = train)
mean((Auto$mpg - predict (lm_fit3, Auto))[-train]^2)
```

    ## [1] 20.43036

## Leave-One-Out Cross-Validation

We can perform LOOCV automatically for any GLM using `glm()` and
`cv.glm()` (part of the `boot` library). We’ll create a GLM of `mpg`
from `horsepower`.

``` r
glm_fit <- glm(mpg ~ horsepower, data = Auto)
coef(glm_fit)
```

    ## (Intercept)  horsepower 
    ##  39.9358610  -0.1578447

``` r
library(boot)
cv_err <- cv.glm(Auto, glm_fit)
cv_err$delta
```

    ## [1] 24.23151 24.23114

This first value is the result of LOOCV given as the first equation in
Chapter 5.1. The second value is the “adjusted cross validation
estimate, … designed to compensate for the bias introduced by not using
\[LOOCV\]” (documentation).

We can repeat this procedure for more complex polynomial fits, using a
for loop to loop through polynomial degrees 1 through 10 to fit
different GLMs:

``` r
cv_error <- rep(0, 10)
for(i in 1:10){
    glm_fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
    cv_error[i] <- cv.glm(Auto, glm_fit)$delta[1]
}
cv_error
```

    ##  [1] 24.23151 19.24821 19.33498 19.42443 19.03321 18.97864 18.83305 18.96115
    ##  [9] 19.06863 19.49093

This takes a significant amount of time, as we are fitting 392 models
for each of the 10 different polynomial degrees, then calculating the
MSE for each of those models. Here, we again see a drop in MSE going
from a linear to a quadratic fit, but no real difference for higher
order polynomials.

Now we’ll take a look at how we can speed this process up.

## k-fold Cross-Validation

We can use the same `cv.glm()` function to implement k-fold CV. We’ll
use k=10, meaning that for each polynomial degree, we will only need to
fit 10 models, instead of 392. This makes things *much* faster.

``` r
set.seed(17)
cv_error_10 <- rep(0, 10)
for(i in 1:10) {
    glm_fit <- glm(mpg ~ poly(horsepower, i), data = Auto)
    cv_error_10[i] <- cv.glm(Auto, glm_fit, K=10)$delta[1]
}
cv_error_10
```

    ##  [1] 24.27207 19.26909 19.34805 19.29496 19.03198 18.89781 19.12061 19.14666
    ##  [9] 18.87013 20.95520

Very similarly to LOOCV, we see little to no evidence that a
higher-order polynomial than 2 is at all useful in predicting `mpg` from
`horsepower`.

With k-fold CV, the two numbers in the `delta` object of the `cv.glm()`
differ slighly, because the second number is “adjusted” for the fact
that it’s not LOOCV.

## The Bootstrap

We’ll use the same example we looked at in Section 5.2, as well as
estimating the accuracy of the linear regression model on the `Auto`
data set.

Using the same definition of alpha that we saw in Section 5.2, we can
create a function which returns alpha based on a given data set and
index range. Then, we can sample randomly from the `Portfolio` data set
and perform a bootstrap analysis using the data set and the alpha
function we created.

``` r
alpha_fn <- function(data, index) {
    X <- data$X[index]
    Y <- data$Y[index]
    return(
        (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2*cov(X, Y))
    )
}

alpha_fn(Portfolio, 1:100)
```

    ## [1] 0.5758321

``` r
set.seed(7)
alpha_fn(Portfolio, sample(1:100, 100, replace = T))
```

    ## [1] 0.5385326

``` r
boot(Portfolio, alpha_fn, R = 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Portfolio, statistic = alpha_fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##      original       bias    std. error
    ## t1* 0.5758321 0.0007959475  0.08969074

The bootstrap analysis gives the original estimate for alpha, and that
the estimate of the standard error is 0.08969.

Let’s take a look at another example. If we want to estimate the
accuracy of a linear regression model, we can use the bootstrap method
to estimate the standard errors of the coefficients. To do this, we can
create a function which produces

``` r
lm(mpg ~ horsepower, data = Auto) %>% coef()
```

    ## (Intercept)  horsepower 
    ##  39.9358610  -0.1578447

``` r
boot_fn <- function(data, index) {
    coef(lm(mpg ~ horsepower, data = data, subset = index))
}
boot_fn(Auto, 1:392)
```

    ## (Intercept)  horsepower 
    ##  39.9358610  -0.1578447

Let’s give two examples of bootstrapping this data set to estimate the
standard error, in order to show that the estimates differ slightly.

``` r
set.seed(1)
boot_fn(Auto, sample(392, 392, replace = T))
```

    ## (Intercept)  horsepower 
    ##  40.3404517  -0.1634868

``` r
boot_fn(Auto, sample(392, 392, replace = T))
```

    ## (Intercept)  horsepower 
    ##  40.1186906  -0.1577063

Now we’ll compute the standard errors of 1000 bootstrap estimates to
approximate the standard error of the coefficients:

``` r
set.seed(1)
boot(Auto, boot_fn, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Auto, statistic = boot_fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##       original        bias    std. error
    ## t1* 39.9358610  0.0553942585 0.843931305
    ## t2* -0.1578447 -0.0006285291 0.007367396

This tells us, then, that the estimate of the standard error for the
intercept on the linear regression model is 0.844, and the estimate of
the standard error for the coefficient on the linear regression model is
0.0074. These values can also given by the summary of the linear model:

``` r
summary(lm(mpg ~ horsepower, data = Auto))$coef
```

    ##               Estimate  Std. Error   t value      Pr(>|t|)
    ## (Intercept) 39.9358610 0.717498656  55.65984 1.220362e-187
    ## horsepower  -0.1578447 0.006445501 -24.48914  7.031989e-81

We can note that the values given here are different than those acquired
using bootstrap, which is interesting. It turns out that the SE values
obtained using the standard equations in the `lm()` command rely on some
assumptions about X and Y. So, these bootstrap estimates are likely more
accurate than the estimates given in the `summary()` of the linear model
in `R`.

Let’s also perform a bootstrap to a quadratic fit of mpg on horsepower:

``` r
boot_fn <- function(data, index) {
    coef(
        lm(mpg ~ horsepower + I(horsepower^2), 
           data = data, subset = index)
    )
}
set.seed(1)
boot(Auto, boot_fn, 1000)
```

    ## 
    ## ORDINARY NONPARAMETRIC BOOTSTRAP
    ## 
    ## 
    ## Call:
    ## boot(data = Auto, statistic = boot_fn, R = 1000)
    ## 
    ## 
    ## Bootstrap Statistics :
    ##         original        bias     std. error
    ## t1* 56.900099702  3.511640e-02 2.0300222526
    ## t2* -0.466189630 -7.080834e-04 0.0324241984
    ## t3*  0.001230536  2.840324e-06 0.0001172164

``` r
summary(
    lm(
        mpg ~ horsepower + I(horsepower^2), 
        data = Auto
    )
)$coef
```

    ##                     Estimate   Std. Error   t value      Pr(>|t|)
    ## (Intercept)     56.900099702 1.8004268063  31.60367 1.740911e-109
    ## horsepower      -0.466189630 0.0311246171 -14.97816  2.289429e-40
    ## I(horsepower^2)  0.001230536 0.0001220759  10.08009  2.196340e-21

We can clearly see here that the standard errors for the `t3` term, or
the quadratic term, are very low. Additionally, the standard errors are
very similar between the bootstrap and the standard formula, but we
assume that the bootstrap estimates are more accurate due to not
requiring assumptions about the behavior of the variables.
