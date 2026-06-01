# A separable target on non-square data (200 x 3). The non-square shape doubles
# as an orientation check: a transposed design matrix at the FFI boundary would
# either error or produce garbage predictions.
make_data <- function(n = 200, seed = 1) {
  set.seed(seed)
  x <- matrix(stats::runif(n * 3, -2, 2), ncol = 3,
              dimnames = list(NULL, c("a", "b", "c")))
  y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + stats::rnorm(n, sd = 0.1)
  list(x = x, y = y)
}

test_that("tsl() fits and returns a well-formed object", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L)

  expect_s3_class(fit, "tsl")
  expect_equal(fit$n_features, 3L)
  expect_equal(fit$n_obs, 200L)
  expect_equal(fit$feature_names, c("a", "b", "c"))
  expect_length(fit$residuals, 200L)
  expect_length(fit$y_hat, 200L)
  expect_true(is.finite(fit$err))
  expect_true(all(is.finite(fit$y_hat)))
})

test_that("predict() returns finite predictions of the right length", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 8L, seed = 123L, verbosity = 0L)

  te <- make_data(n = 50, seed = 2)
  p <- predict(fit, te$x)

  expect_length(p, 50L)
  expect_true(all(is.finite(p)))
  # The target is largely separable, so predictions should track the truth.
  expect_gt(stats::cor(p, te$y), 0.8)
})

test_that("fit is reproducible for a fixed seed", {
  d <- make_data()
  f1 <- tsl(d$x, d$y, epochs = 8L, seed = 7L, verbosity = 0L)
  f2 <- tsl(d$x, d$y, epochs = 8L, seed = 7L, verbosity = 0L)
  expect_equal(predict(f1, d$x), predict(f2, d$x))
})

test_that("training fit explains most variance", {
  d <- make_data()
  fit <- tsl(d$x, d$y, epochs = 10L, seed = 123L, verbosity = 0L)
  yhat <- predict(fit, d$x)
  r2 <- 1 - sum((d$y - yhat)^2) / sum((d$y - mean(d$y))^2)
  expect_gt(r2, 0.7)
})

test_that("invalid strategy names are rejected", {
  d <- make_data(n = 20)
  expect_error(tsl(d$x, d$y, split_strategy = "nope"))
  expect_error(tsl(d$x, d$y, refinement_strategy = "nope"))
})

test_that("dimension mismatch in predict() is caught", {
  d <- make_data(n = 30)
  fit <- tsl(d$x, d$y, epochs = 3L, seed = 1L, verbosity = 0L)
  expect_error(predict(fit, d$x[, 1:2]), "features")
})
