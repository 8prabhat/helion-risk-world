from __future__ import annotations


class ExecutionLogBuilder:
    """Build execution-state features from own fill logs (V2 calibration).

    SRP: execution features only.
    """

    def build(self) -> object:
        raise NotImplementedError("ExecutionLogBuilder.build — SPEC.md §15 (V2)")
