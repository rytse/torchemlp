import sys

sys.path.append("../")

from torchemlp.reps import V
from torchemlp.groups import SO, O, SO13

print(f"V + V = {V + V}")
print(f"V * V = {V * V}")
print(f"V.T = {V.T}")
print(f"(V + V.T) * (V * V.T + V) = {(V + V.T) * (V * V.T + V)}")

print(f"5 * V * 2 = {5 * V * 2}")
print(f"2 * (V ** 3) = {2 * (V ** 3)}")

print(f"2 * V(O(4)) ** 3 = {2 * V(O(4)) ** 3}")
print(f"(2 * V ** 3)(O(4))= {(2 * V ** 3)(O(4))}")

print(f"V(SO(3)).T + V(SO(3)) = {V(SO(3)).T + V(SO(3))}")
print(f"V(SO13()).T+V(SO13()) = {V(SO13()).T+V(SO13())}")
