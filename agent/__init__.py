from __future__ import annotations


def run_agent(*args, **kwargs):
	from agent.run import run_agent as _run_agent

	return _run_agent(*args, **kwargs)


__all__ = ["run_agent"]
