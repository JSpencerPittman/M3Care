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

    def track(self, grad_inp: tuple[Tensor], grad_out: tuple[Tensor]):
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


class GradientTracker(object):
    def __init__(self, model: nn.Module):
        self.tracked_modules: list[str] = []
        self.trackers: dict[str, SubmoduleGradientTracker] = {}

        for name, module in model.named_modules():
            self.tracked_modules.append(name)
            module.register_full_backward_hook(self._create_backward_hook(name))

    def _create_backward_hook(self, name: str):
        def hook(module: nn.Module, grad_inp: tuple[Tensor], grad_out: tuple[Tensor]):
            if name not in self.trackers:
                self.trackers[name] = SubmoduleGradientTracker(name,
                                                               len(grad_inp),
                                                               len(grad_out))
            self.trackers[name].track(grad_inp, grad_out)
        return hook
