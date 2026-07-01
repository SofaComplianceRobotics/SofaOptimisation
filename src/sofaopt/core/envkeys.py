"""The environment-variable contract between the optimizer and a scene process.

These keys are set by the parallel runner on each ``runSofa`` subprocess and
read back by the scene via :mod:`sofaopt.scene`. Defined once here so both
sides stay in sync.
"""

# Set on every scene subprocess by the runner:
TRIAL_STATE_PATH = "OPT_TRIAL_STATE_PATH"  # where the scene writes its score
PARAMS_PATH = "OPT_PARAMS_PATH"            # JSON file with this trial's sampled params
RUN_SLOT = "OPT_RUN_SLOT"                  # 1-based index of this run in the trial
TEST_NAME = "OPT_TEST_NAME"                # which TestSpec is being evaluated
TEST_RUN_INDEX = "OPT_TEST_RUN_INDEX"      # 1-based repeat index within the test
TEST_RUN_TOTAL = "OPT_TEST_RUN_TOTAL"      # number of repeats for the test
GEN = "OPT_GEN"                            # generation number
TRIAL = "OPT_TRIAL"                        # trial number within the generation
RUN = "OPT_RUN"                            # global run number within the trial

# Selection/weights forwarded so any child can reconstruct them (optional use):
SELECTED_TESTS = "OPT_SELECTED_TESTS"      # comma-separated test names
TEST_WEIGHTS = "OPT_TEST_WEIGHTS"          # JSON {name: int percent}
GATED_TESTS = "OPT_GATED_TESTS"            # comma-separated gated test names

# Optimizer-setting overrides set by the dashboard's Run button so a launched
# headless run honors the UI choices without editing the project (read once by
# run_optimization → applied over the project fields before build_study):
SAMPLER = "OPT_SAMPLER"                    # "cmaes" | "gp" | "tpe" | "random"
CMAES_MARGIN = "OPT_CMAES_MARGIN"          # "1"/"true" to enable CMA-ES-with-Margin
SEED_SAMPLER = "OPT_SEED_SAMPLER"          # "random" | "sobol" (initial-design)
N_PARALLEL = "OPT_N_PARALLEL"              # int override for project.n_parallel
N_GENERATIONS = "OPT_N_GENERATIONS"        # int override for project.n_generations
