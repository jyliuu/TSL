# Composite local-interpretation dashboard for fitted TSL models, in the flat
# aesthetic. For each query point, three sub-plots (per-stage net contribution,
# per-feature backbone share, signed per-feature tilt) are built from
# tsl_local() and composed in a row with patchwork; when patchwork is not
# installed the per-point tsl_local() results are returned instead.

# Coerce `points` (a single numeric vector, a list of vectors, or a matrix with
# one point per row) to a list of numeric vectors.
.tsl_as_point_list <- function(points) {
  if (is.list(points)) {
    return(lapply(points, as.numeric))
  }
  if (is.matrix(points)) {
    return(lapply(seq_len(nrow(points)), function(i) as.numeric(points[i, ])))
  }
  list(as.numeric(points))
}

#' Plot the local-interpretation dashboard
#'
#' Composes a per-point decomposition of one or more TSL predictions. Each query
#' point becomes a row of three panels: a stage-contribution view (the positive
#' branch in orange, the negative branch in blue, and the net contribution as a
#' dark point, with stages ordered by absolute net effect); a backbone-share bar
#' chart (each feature's share of `|log b_j|` summed over stages); and a
#' signed-tilt bar chart (each feature's total tilt, coloured by sign).
#'
#' @param object A fitted model of class `"tsl"` from [tsl()].
#' @param points A single numeric vector of length `n_features`, a list of such
#'   vectors, or a matrix with one point per row.
#' @param titles Per-point panel titles. Defaults to `"Point 1"`, `"Point 2"`, ...
#' @param top_k_features Number of features to keep in the backbone-share and
#'   signed-tilt panels (the remainder are lumped into `"Other"` for the
#'   backbone share).
#' @return A patchwork object (or a list of ggplots if patchwork is not
#'   installed). The per-point [tsl_local()] results are attached as the
#'   `"tsl_data"` attribute (see [tsl_plot_data()]).
#' @seealso [tsl_local()]
#' @examples
#' set.seed(1)
#' x <- matrix(runif(200 * 3, -2, 2), ncol = 3,
#'             dimnames = list(NULL, c("a", "b", "c")))
#' y <- 2 * x[, 1] - x[, 2] + 0.5 * x[, 3] + rnorm(200, sd = 0.1)
#' fit <- tsl(x, y, epochs = 5L, n_trees = 5L, verbosity = 0L)
#' plot_local_interpretation(fit, x[1, ])
#' plot_local_interpretation(fit, x[1:2, ], titles = c("A", "B"))
#' @export
plot_local_interpretation <- function(object, points, titles = NULL,
                                      top_k_features = 3L) {
  .tsl_check_model(object)
  points <- .tsl_as_point_list(points)
  if (is.null(titles)) titles <- paste("Point", seq_along(points))

  build <- function(ex, title) {
    # Stage contributions, largest |net| at the top (ggplot y runs bottom-up).
    st <- ex$stages
    ord <- order(abs(st$net))
    st$stage <- factor(as.character(st$stage),
                       levels = as.character(st$stage)[ord])
    a <- ggplot(st) +
      geom_vline(xintercept = 0, colour = .tsl_tokens$faint,
                 linetype = "dashed", linewidth = 0.4) +
      geom_col(aes(fpos, stage), fill = .tsl_tokens$pos,
               alpha = 0.85, width = 0.6) +
      geom_col(aes(fneg, stage), fill = .tsl_tokens$neg,
               alpha = 0.85, width = 0.6) +
      geom_point(aes(net, stage), colour = .tsl_tokens$ink, size = 2) +
      labs(title = title,
           subtitle = sprintf("prediction %.3f", ex$total_prediction),
           x = "contribution", y = NULL) +
      theme_flat() +
      theme(panel.grid.major.y = element_blank())

    # Backbone share: per-feature share of |log b_j| summed over stages.
    lb <- abs(log(pmax(ex$feature_backbone, 1e-12)))
    share <- colSums(lb)
    tot <- sum(share)
    share <- if (tot > 0) share / tot else share * 0
    bb <- data.frame(feature = ex$feature_names, share = share,
                     stringsAsFactors = FALSE)
    bb <- bb[order(bb$share, decreasing = TRUE), , drop = FALSE]
    if (nrow(bb) > top_k_features) {
      tail_share <- sum(bb$share[seq.int(top_k_features + 1L, nrow(bb))])
      bb <- rbind(bb[seq_len(top_k_features), ],
                  data.frame(feature = "Other", share = tail_share,
                             stringsAsFactors = FALSE))
    }
    b <- ggplot(bb) +
      geom_col(aes(share, reorder(feature, share)),
               fill = .tsl_tokens$accent, width = 0.7) +
      labs(title = "Backbone share", x = "share", y = NULL) +
      theme_flat()

    # Signed tilt: per-feature total tilt over stages, top-k by magnitude.
    tilt <- colSums(ex$feature_tilt)
    tl <- data.frame(feature = ex$feature_names, tilt = tilt,
                     stringsAsFactors = FALSE)
    tl <- tl[order(abs(tl$tilt), decreasing = TRUE), , drop = FALSE]
    tl <- utils::head(tl, top_k_features)
    c_plot <- ggplot(tl) +
      geom_col(aes(tilt, reorder(feature, tilt), fill = tilt > 0),
               width = 0.7) +
      scale_fill_manual(values = c(`FALSE` = .tsl_tokens$neg,
                                   `TRUE` = .tsl_tokens$pos), guide = "none") +
      geom_vline(xintercept = 0, colour = .tsl_tokens$faint,
                 linetype = "dashed") +
      labs(title = "Signed tilt", x = "tilt", y = NULL) +
      theme_flat()

    list(stage = a, backbone = b, tilt = c_plot)
  }

  ex_list <- lapply(points, function(pt) tsl_local(object, pt))
  sub <- Map(build, ex_list, titles)

  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("Install 'patchwork' for the composed dashboard; ",
            "returning the per-point explanations.")
    return(ex_list)
  }

  rows <- lapply(sub, function(s) {
    patchwork::wrap_plots(s$stage, s$backbone, s$tilt, nrow = 1)
  })
  out <- patchwork::wrap_plots(rows, ncol = 1) +
    patchwork::plot_annotation(title = "Local interpretation")
  attr(out, "tsl_data") <- ex_list
  out
}
