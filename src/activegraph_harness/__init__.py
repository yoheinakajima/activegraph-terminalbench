"""activegraph-harness: a Harbor agent for Terminal-Bench 2.0 whose memory
and logging substrate is an ActiveGraph event store.

The log is the agent: every step of every trial is appended as an event to
a per-trial ActiveGraph store, and the graph is the deterministic projection
of the run.
"""

__version__ = "0.1.0"
