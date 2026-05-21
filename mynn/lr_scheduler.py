from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass

class ConstantLR(scheduler):
    def __init__(self, optimizer) -> None:
        super().__init__(optimizer)

    def step(self) -> None:
        self.step_count += 1

class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        milestones: list of integers. Must be increasing. (e.g., [800, 2400, 4000])
        gamma: multiplicative factor of learning rate decay.
        """
        super().__init__(optimizer)
        # 将 milestones 转换为 set，这样在 step() 中用 in 判断时复杂度为 O(1)，效率更高
        self.milestones = set(milestones)
        self.gamma = gamma

    def step(self) -> None:
        # 步数全局累加
        self.step_count += 1
        
        # 当当前的全局步数触发了设定的里程碑时，衰减学习率
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(scheduler):
    pass