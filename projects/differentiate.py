import numpy as np


def differentiate(u: np.ndarray, dt: float) -> np.ndarray:
    du = []
    for i in range(1, len(u)):
        du.append(float((u[i] - u[i-1])/dt))
    return du

def differentiate_vector(u: np.ndarray, dt: float) -> np.ndarray:
    du = np.diff(u) / dt
    return du

def test_differentiate():
    t = np.linspace(0, 1, 10)
    dt = t[1] - t[0]
    u = t**2
    du1 = differentiate(u, dt)
    du2 = differentiate_vector(u, dt)
    print(du1 == du2)
    assert np.allclose(du1, du2)

if __name__ == '__main__':
    test_differentiate()
    