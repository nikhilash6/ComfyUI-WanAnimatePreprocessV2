# -*- coding: utf-8 -*-
"""Shared helper: ComfyUI green progress bar + tqdm ETA + Stop-button interrupt.

Two public APIs:

* ``check()`` — call inside any short loop body to surface Stop clicks.
* ``track(iterable, total=None, desc="")`` — drop-in replacement for
  ``range(...)`` / ``enumerate(...)``. Drives the green progress fill on
  the node UI via ``comfy.utils.ProgressBar`` AND a tqdm terminal line
  with ETA, AND polls ``processing_interrupted`` between iterations.

All three integrations degrade to no-ops when ComfyUI / tqdm are absent
(stand-alone tests), so this module is safe to import anywhere.
"""
from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:
    from comfy.model_management import (  # type: ignore
        throw_exception_if_processing_interrupted as _throw,
    )
except Exception:  # noqa: BLE001
    def _throw() -> None:  # type: ignore[no-redef]
        return None

try:
    from comfy.utils import ProgressBar as _ComfyPB  # type: ignore
except Exception:  # noqa: BLE001
    _ComfyPB = None  # type: ignore

try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # noqa: BLE001
    _tqdm = None  # type: ignore


def check() -> None:
    """Raise ``InterruptProcessingException`` if the user clicked Stop."""
    _throw()


def track(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
) -> Iterator[T]:
    """Yield from ``iterable`` while updating UI + terminal progress."""
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

    pbar = None
    if _ComfyPB is not None and total:
        try:
            pbar = _ComfyPB(int(total))
        except Exception:
            pbar = None

    if _tqdm is not None:
        it = _tqdm(iterable, total=total, desc=desc or None, leave=False)
    else:
        it = iterable

    i = 0
    try:
        for item in it:
            _throw()
            yield item
            i += 1
            if pbar is not None:
                try:
                    pbar.update_absolute(i)
                except Exception:
                    pass
    finally:
        if _tqdm is not None and hasattr(it, "close"):
            try:
                it.close()  # type: ignore[union-attr]
            except Exception:
                pass
