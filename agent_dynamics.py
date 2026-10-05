class DynamicAgent:
    """x: fast internal state (tau). rbar: slow average reward rate (tau_slow).
    Leave when x falls below rbar (marginal value theorem)."""
 
    def __init__(self, tau=0.3, tau_slow=0.05, x0=10.0, rbar0=3.0):
        self.tau, self.tau_slow, self.x0 = tau, tau_slow, x0
        self.x, self.rbar, self.n = x0, rbar0, 0
 
    def decide(self, state=None):
        return "leave" if self.n >= 1 and self.x < self.rbar else "stay"
 
    def update(self, r, mode, **kw):
        if mode == "exploit":
            self.x += self.tau * (r - self.x)
            self.n += 1
        else:                       # exploring: reset the fast state
            self.x, self.n = self.x0, 0
        self.rbar += self.tau_slow * (r - self.rbar)
 