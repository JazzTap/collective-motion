import numpy as np
from numpy.linalg import norm

class Sim:
    def __init__(self, ics, pars, T):
        # each boid as (x, y, dx, dy) \in phasespace.
        self.pop = ics.shape[0];  assert( ics.shape[1] == 4)
        self.trace = np.zeros((T, self.pop, 4))

        self.trace[0, :, :] = ics;  self.pars = pars
        self.t = 0; self.T = T
        
    # setter (todo: as iterator)
    def run(self):
        for i in range(self.T - 1): self.step()
            
    def step(self):
        t = self.t # evaluate the next point in time.
        for j in range(self.pop):
            x = self.trace[t,j,:] # for each boid,
            acc = self.plan(t, x, **self.pars) # make a plan,
            self.trace[t+1,j,:] = self.apply(x, acc) # and update trace.
        self.t += 1
        
    # remaining methods have no side effects
        
    # utility (todo: make static)
    def limit(self, vec, mag):
        m = norm(vec); return vec if m < mag else mag * vec / m
    def apply(self, x, acc):
        y = np.copy(x); y[2:4] += acc; y[0:2] += y[2:4]; return y

    # analysis
    def report(self, t):
        return self.trace[t, :, :]
    
    def field(self, t, xs, speed=None):
        # report acceleration at sample points xs, at time t
        n = xs.shape[0]; acc = np.zeros((n, 2))
        
        # VERIFY optional override of speed limit
        pars = dict(self.pars)
        if speed: pars['speed'] = speed
        
        for i in range(n):
            acc[i,:] = self.plan(t, xs[i,:], **pars)
        return np.hstack([xs[:,0:2], acc])
    
    # core
    def plan(self, t, me,
             separate, align, cohere,
             close=3, far=12, steer=.1, speed=1):
        # given 'me = x,y,dx,dy', return my next move
        # as a change in velocity. 't' an evaluated point in time.
        
        them = self.trace[t, :, :] # also contains 'me', but 'me - me = 0'.

        def nbh(r): # neighbors, not on top of me, as boolean vector.
            diffs = (them - me)[:, 0:2]; dists = norm(diffs, axis=1)
            choices = np.logical_and(dists < r, dists > 0)
            return choices

        too_close = nbh(close)
        near_me = nbh(far)

        def if_any(visible, make_vel):
            # todo: fail out of do_s gracefully.
            return make_vel() if np.any(visible) else np.array([0,0])

        def do_separate():
            diffs = (me - them)[too_close, 0:2]; dists = norm(diffs, axis=1)
            weighted = (diffs.T / (dists*dists)).T # == A / (a*a)[:,None]

            vec = np.average(weighted, axis=0)
            return self.limit(vec, steer) # Reynolds steering

        def do_align():
            vec = np.average(them[near_me, 2:4], axis=0)
            return self.limit(vec, steer)

        def do_cohere():
            tar = np.average(them[near_me, 0:2], axis=0)
            vec = tar - me[0:2]
            return self.limit(vec, steer)
        
        acc = separate * if_any(too_close, do_separate) +\
              align * if_any(near_me, do_align) +\
              cohere * if_any(near_me, do_cohere)
        
        # constrain reported acceleration by max speed
        return self.limit(me[2:4] + acc, speed) - me[2:4]