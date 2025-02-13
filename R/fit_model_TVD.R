#' @export
fit_model_TVD <- function(
    data,
    seed = NULL,
    refresh = NULL,
    init = NULL,
    save_latent_dynamics = FALSE,
    output_dir = NULL,
    output_basename = NULL,
    sig_figs = NULL,
    chains = 4,
    parallel_chains = getOption("mc.cores", 1),
    chain_ids = seq_len(chains),
    threads_per_chain = NULL,
    opencl_ids = NULL,
    iter_warmup = 1000,
    iter_sampling = 1000,
    save_warmup = FALSE,
    thin = NULL,
    max_treedepth = NULL,
    adapt_engaged = TRUE,
    adapt_delta = NULL,
    step_size = NULL,
    metric = NULL,
    metric_file = NULL,
    inv_metric = NULL,
    init_buffer = NULL,
    term_buffer = NULL,
    window = NULL,
    fixed_param = FALSE,
    show_messages = TRUE,
    compile = TRUE,
    ...
) {
  
  stan_file <- system.file("stan", "twostate_2.0.stan", package = "mrmrmr", mustWork = TRUE)
  
  model <- cmdstanr::cmdstan_model(stan_file, ...)
  
  m_fit <- model$sample(
    data = data$stan_d,
    seed = seed,
    refresh = refresh,
    init = init,
    save_latent_dynamics = save_latent_dynamics,
    output_dir = output_dir,
    output_basename = output_basename,
    sig_figs = sig_figs,
    chains = chains,
    parallel_chains = parallel_chains,
    chain_ids = chain_ids,
    threads_per_chain = threads_per_chain,
    opencl_ids = opencl_ids,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    save_warmup = save_warmup,
    thin = thin,
    max_treedepth = max_treedepth,
    adapt_engaged = adapt_engaged,
    adapt_delta = adapt_delta,
    step_size = step_size,
    metric = metric,
    metric_file = metric_file,
    inv_metric = inv_metric,
    init_buffer = init_buffer,
    term_buffer = term_buffer,
    window = window,
    fixed_param = fixed_param,
    show_messages = show_messages
  )
  
  # Load all the data and return the whole unserialized fit object
  m_fit$draws() # Do not specify variables or inc_warmup.
  try(m_fit$sampler_diagnostics(), silent = TRUE)
  try(m_fit$init(), silent = TRUE)
  try(m_fit$profiles(), silent = TRUE)
  
  list(m_fit = m_fit, data = data)
}


