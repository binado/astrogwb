"""Reference importance-weights models packaged as merger-rate/log-weight callbacks."""

from .bns_madau_dickinson_modified_propagation import (
    make_merger_rate_and_log_weights_fn,
)

__all__ = ["make_merger_rate_and_log_weights_fn"]
