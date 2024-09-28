from itertools import product
from typing import Optional

from util.grad_track import GradientTracker, SubmoduleGradientTracker

ShrinkEntry = tuple[str, int, int, float]


class SubmoduleGradientShrinkage(object):
    def __init__(self, name: str, sub_gt: SubmoduleGradientTracker):
        self.name = name
        self.shrinkage: dict[int, dict[int, float]] = {}  # Index(inp, out, step)

        for inp_idx, out_idx in product(range(sub_gt.num_inp), range(sub_gt.num_out)):
            # Do not calculate shrinkage on None entries
            if sub_gt.in_grad_is_none[inp_idx] or sub_gt.out_grad_is_none[out_idx]:
                continue

            # Calculate shrinkage (remember input comes AFTER output gradient)
            shrink = [g_out / g_in
                      for g_in, g_out
                      in zip(sub_gt.grad_in_norms[inp_idx],
                             sub_gt.grad_out_norms[out_idx],
                             strict=True)]

            if inp_idx not in self.shrinkage:
                self.shrinkage[inp_idx] = {out_idx: shrink}
            else:
                self.shrinkage[inp_idx][out_idx] = shrink

    def grab(self, largest_only: bool = False) -> list[ShrinkEntry]:
        aggregated: list[ShrinkEntry] = []

        for inp_idx, v in self.shrinkage.items():
            for out_idx, shrink in v.items():
                aggregated.append((self.name, inp_idx, out_idx, shrink))

        if largest_only:
            sorted(aggregated,
                   key=lambda se: se[-1],
                   reverse=True)
            return [aggregated[0]]
        else:
            return aggregated


class GradientShrinkage(object):
    def __init__(self, gt: GradientTracker):
        self.shrinkages = {name: SubmoduleGradientShrinkage(name, sub_gt)
                           for name, sub_gt
                           in gt.trackers.items()}

    def grab(self,
             submodule_name: Optional[str] = None,
             largest_per_submodule: bool = False,
             sort_results: bool = False):
        if submodule_name is not None:
            selection = self.shrinkages[submodule_name].grab(largest_per_submodule)
        else:
            selection = []
            for sub_shrink in self.shrinkages.values():
                selection += sub_shrink.grab(largest_per_submodule)

        if sort_results:
            selection = sorted(selection,
                               key=lambda se: se[-1],
                               reverse=True)

        return selection
