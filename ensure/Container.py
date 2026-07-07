"""ENSURE container variant.

Extends ``dynamic_pool.Container`` with a ``protected`` flag used by FnScale's
square-root staffing rule: protected containers form the warm-pool buffer and
must NOT be evicted by the idle timer (the paper keeps them warm via periodic
heartbeats — same effect, simpler implementation).
"""
from __future__ import annotations

import os
import sys

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from dynamic_pool.Container import Container


class EnsureContainer(Container):
    def __init__(self, env, system, server, resource_info, idle_timeout,
                 is_cold_start=True, protected=False):
        super().__init__(env, system, server, resource_info, idle_timeout,
                         is_cold_start=is_cold_start)
        self.protected = protected

    def release_request(self):
        super().release_request()
        # FnScale buffer containers never time out — cancel the timer that the
        # base release_request just started.
        if self.protected and self.idle_timer_proc is not None:
            if self.idle_timer_proc.is_alive:
                try:
                    self.idle_timer_proc.interrupt()
                except RuntimeError:
                    pass
            self.idle_timer_proc = None
