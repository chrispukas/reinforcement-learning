import numpy.random as rand
from typing import Dict, Optional, List, Any


class Arm:
    def __init__(self,
                 mean: Optional[float] = 0,
                 std: Optional[float] = 1,
                 ) -> None:
        self.mean: float = mean
        self.std: float = std
    
    def get(self):
        return rand.normal(
            loc=self.mean,
            scale=self.std,
            size=(1,)
        )

class KBandit:
    def __init__(self, 
                 num: Optional[int] = 5,
                 discount_factor: Optional[float] = 0.5,
                 epsilon: Optional[float] = 0.1
                 ) -> None:
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.q_vals: List[float] = [0] * num
        self.arms: List[Arm] = [None] * num
        for i in range(num):
            self.arms[i] = Arm(
                mean=0,
                std=1
            )

    def timestep(self) -> List[float]:
        # Argmax Q (index is the action value)
        r: float = rand.random((1,))[0]
        if r > self.epsilon:
            q_pred: float = max(self.q_vals)
            idx: int = self.q_vals.index(q_pred)
        else:
            idx: int = rand.randint(0, len(self.q_vals))
            q_pred: float = self.q_vals[idx]

        reward: float = self.pull(idx=idx)

        q_new: float = q_pred + self.discount_factor * (reward - q_pred) # Standard update function, with a fixed discount factor -> Non-stationary problem simulated by gaussain distribution
        self.q_vals[idx] = q_new
        return self.q_vals

    def pull(self, 
             idx: int
             ) -> None: 
        arm: Arm = self.arms[idx]

        if arm is None:
            return None
        return self.arms[idx].get()[0]


kbandit: KBandit = KBandit(num=5)

for i in range(500000):
    ts = kbandit.timestep()
    print(ts)
