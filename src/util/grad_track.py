from enum import Enum
from typing import Optional

from torch import Tensor, nn


class SubmoduleGradientTracker(object):
    def __init__(self, name: str, num_inp: int, num_out: int):
        self.name = name
        self.num_inp = num_inp
        self.num_out = num_out

        self.grad_in_norms: list[list[float]] = [[]] * num_inp  # Index(Inp, Step)
        self.grad_out_norms: list[list[float]] = [[]] * num_out  # Index(Out, Step)

        self.in_grad_is_none: list[bool] = [False] * num_inp
        self.out_grad_is_none: list[bool] = [False] * num_out

    def track(self,
              grad_inp: tuple[Optional[Tensor]],
              grad_out: tuple[Optional[Tensor]]):
        assert self.num_inp == len(grad_inp)
        assert self.num_out == len(grad_out)

        for inp_idx, grad in enumerate(grad_inp):
            if grad is not None:
                self.grad_in_norms[inp_idx].append(grad.norm().item())
            else:
                self.in_grad_is_none[inp_idx] = True

        for out_idx, grad in enumerate(grad_out):
            if grad is not None:
                self.grad_out_norms[out_idx].append(grad.norm().item())
            else:
                self.out_grad_is_none[out_idx] = True


class ParamGradientTracker(object):
    def __init__(self, name: str):
        self.name = name
        self.grad_norms: list[float] = []
        self.is_none = False

    def track(self, grad: Optional[Tensor]):
        if grad is None:
            self.is_none = True
            self.grad_norms.append(None)
        else:
            self.grad_norms.append(grad.norm().item())


class GradientTracker(object):
    def __init__(self, model: nn.Module):
        self.tracked_modules: list[str] = []
        self.tracked_params: list[str] = []

        self.module_trackers: dict[str, SubmoduleGradientTracker] = {}
        self.param_trackers: dict[str, ParamGradientTracker] = {}

        for name, module in model.named_modules():
            self.tracked_modules.append(name)
            module.register_full_backward_hook(self._create_module_backward_hook(name))

        for name, param in model.named_parameters():
            self.tracked_params.append(name)
            param.register_hook(self._create_parameter_backward_hook(name))

    def _create_module_backward_hook(self, name: str):
        def hook(module: nn.Module, grad_inp: tuple[Tensor], grad_out: tuple[Tensor]):
            if name not in self.module_trackers:
                self.module_trackers[name] = SubmoduleGradientTracker(name,
                                                                      len(grad_inp),
                                                                      len(grad_out))
            self.module_trackers[name].track(grad_inp, grad_out)
        return hook

    def _create_parameter_backward_hook(self, name: str):
        def hook(grad: Optional[Tensor]):
            if name not in self.param_trackers:
                self.param_trackers[name] = ParamGradientTracker(name)
            self.param_trackers[name].track(grad)
        return hook


class GradientFlowAnalyzer(object):
    class StepType(Enum):
        ModuleIn = 0
        ModuleOut = 1
        Parameter = 2

    Flow = list[tuple[StepType, float]]

    def __init__(self, gt: GradientTracker):
        self.gt = gt

    def flow(self,
             order: list[str | tuple[str, int, int]],
             steps: int | list[int]
             ) -> dict[int, Flow]:
        flow: dict[int, GradientFlowAnalyzer.Flow] = {}

        if isinstance(steps, int):
            steps = [steps]

        for step in steps:
            flow[step] = []
            for part in order:
                if isinstance(part, tuple) or (part in self.gt.tracked_modules):
                    if isinstance(part, str):
                        name, in_idx, out_idx = part, 0, 0
                    else:
                        name, in_idx, out_idx = part

                    tracker = self.gt.module_trackers[name]
                    if not tracker.out_grad_is_none[out_idx]:
                        flow[step].append((name,
                                           GradientFlowAnalyzer.StepType.ModuleOut,
                                           tracker.grad_out_norms[out_idx][step]))
                    if not tracker.in_grad_is_none[in_idx]:
                        flow[step].append((name,
                                           GradientFlowAnalyzer.StepType.ModuleIn,
                                           tracker.grad_in_norms[in_idx][step]))
                else:
                    name = part
                    tracker = self.gt.param_trackers[name]
                    if not tracker.is_none:
                        flow[step].append((name,
                                           GradientFlowAnalyzer.StepType.Parameter,
                                           tracker.grad_norms[step]))
        return flow

    @staticmethod
    def shrinkage(flows: dict[int, Flow]):
        shrinkages: dict[int, list[tuple[str, float]]] = {}

        for step, flow in flows.items():
            shrinkages[step] = GradientFlowAnalyzer._shrinkage_flow(flow)
        return shrinkages

    @staticmethod
    def _shrinkage_flow(flow: Flow) -> list[tuple[str, float]]:
        shrinkages: list[tuple[str, float]] = []

        for i, o in zip(flow[:-1], flow[1:]):
            label = f"{i[0]} -> {o[0]}"
            shrinkages.append((label, f"{(i[-1]/o[-1]):.3f}"))

        return shrinkages


def pretty_flow(flow: GradientFlowAnalyzer.Flow):
    result = ""
    for idx, entry in enumerate(flow):
        match entry[1]:
            case GradientFlowAnalyzer.StepType.ModuleIn:
                if idx <= 0 or flow[idx-1][1] != GradientFlowAnalyzer.StepType.ModuleOut:
                    result += f"{entry[0]}(I): {entry[2]:e}\n"
            case GradientFlowAnalyzer.StepType.ModuleOut:
                if idx < len(flow) - 1 and flow[idx+1][1] == GradientFlowAnalyzer.StepType.ModuleIn:
                    in_grad = flow[idx+1][2]
                    result += f"{entry[0]}(O->I): {entry[2]:e} -> {in_grad:e}\n"
                else:
                    result += f"{entry[0]}(O): {entry[2]:e}\n"
            case GradientFlowAnalyzer.StepType.Parameter:
                result += f"{entry[0]}(P): {entry[2]:e}\n"
    return result