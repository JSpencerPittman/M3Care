from itertools import product
from typing import Optional
from collections import namedtuple
from util.grad_track import GradientTracker, SubmoduleGradientTracker
from copy import copy


ShrinkEntry = namedtuple('ShrinkEntry', ['name', 'in_idx', 'out_idx', 'value'])
ShrinkEntries = list[ShrinkEntry]


class ShrinkTable(object):
    def __init__(self):
        self.table: list[ShrinkEntries] = []
        self.num_steps: int = 0
        self._sorted = False

    def save_entry(self, step: int, entry: ShrinkEntry):
        if self.num_steps <= step:
            self.table += [[]] * ((step+1)-self.num_steps)
            self.num_steps = step+1

        self._sorted = False

        self.table[step].append(entry)

    def sort(self):
        if self._sorted:
            return

        for step_idx, shrinks in enumerate(self.table):
            self.table[step_idx] = sorted(shrinks,
                                          key=lambda se: se.value,
                                          reverse=True)

        self._sorted = True

    def top(self,
            n: int = 1,
            steps: Optional[int | slice] = None) -> list[ShrinkEntries]:
        self.sort()
        if isinstance(steps, int):
            steps = slice(steps, steps+1)
        elif steps is None:
            steps = slice(0, self.num_steps)

        filt = copy(self.table[slice(0, self.num_steps) if steps is None else steps]) 

        for step_idx, shrinks in enumerate(filt):
            filt[step_idx] = shrinks[:n]

        return filt

    def __iadd__(self, other):
        for step_idx, shrink in enumerate(other.table):
            if step_idx < self.num_steps:
                self.table[step_idx] += shrink
            else:
                self.table.append(shrink)
                self.num_steps += 1

        self._sorted = False
        return self


class SubmoduleGradientShrinkage(object):
    def __init__(self, name: str, sub_gt: SubmoduleGradientTracker):
        self.name = name
        self.shrink = ShrinkTable()

        for in_idx, out_idx in product(range(sub_gt.num_inp), range(sub_gt.num_out)):
            # Do not calculate shrinkage on None entries
            if sub_gt.in_grad_is_none[in_idx] or sub_gt.out_grad_is_none[out_idx]:
                continue

            # Calculate shrinkage (remember input comes AFTER output gradient)
            shrink = [g_out / g_in
                      for g_in, g_out
                      in zip(sub_gt.grad_in_norms[in_idx],
                             sub_gt.grad_out_norms[out_idx],
                             strict=True)]

            for step_idx, v in enumerate(shrink):
                self.shrink.save_entry(step_idx, ShrinkEntry(name, in_idx, out_idx, v))

    def grab(self) -> ShrinkTable:
        return self.shrink


class GradientShrinkage(object):
    def __init__(self, gt: GradientTracker):
        self.shrinkages = {name: SubmoduleGradientShrinkage(name, sub_gt)
                           for name, sub_gt
                           in gt.trackers.items()}

    def grab(self,
             submodule_name: Optional[str] = None) -> ShrinkTable:
        if submodule_name is not None:
            selection = self.shrinkages[submodule_name]
        else:
            selection = ShrinkTable()
            for sub_shrink in self.shrinkages.values():
                selection += sub_shrink.grab()

        return selection
