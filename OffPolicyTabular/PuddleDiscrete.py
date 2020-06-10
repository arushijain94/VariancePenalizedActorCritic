
import numpy as np
import gym
from gym import core, spaces
from gym.envs.registration import register




class PuddleD(gym.Env):
    def __init__(self):

        layout = """\
wwwwwwwwwwww
w          w
w          w
w          w
w   ffff   w
w   ffff   w
w   ffff   w
w   ffff   w
w          w
w          w
w          w
wwwwwwwwwwww
"""
        """
        Direction:
        0:U
        1:D
        2:L
        3:R
        4:UR
        5:DR
        6:DL
        7:UL
        Deterministic Actions
        Introducing variable rewards in f "frozen"/ "slippery" state in range U[-8, 8] where expected value is zero as another states
        Reward U[-8, 8] would be awarded when the agent going to the frozen state
        Reward for Goal state : 50
        Reward 0 would be awarded when agent going to normal state
        """

        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        self.frozen = np.array([list(map(lambda c: 1 if c=='f' else 0, line)) for line in layout.splitlines()])


        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))

        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1)), np.array((-1,1)), np.array((1,1)), np.array((1,-1)), np.array((-1,-1))]

        self.tostate = {}
        statenum = 0
        for i in range(12):
            for j in range(12):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k,v in self.tostate.items()}
        
        self.goal = 9
        self.init_states = list(range(self.observation_space.n))
        self.init_states.remove(self.goal)

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):
        state = np.random.choice(self.init_states)
        self.currentcell = self.tocell[state]
        return state

    def step(self, action):
        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/3,
        the agent moves instead in one of the other three directions, each with 1/9 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        We consider a case in which rewards are zero on all state transitions.
        """
        reward = 0
        if np.random.uniform() < 1/5.:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[np.random.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if self.frozen[self.currentcell]:
            reward = np.random.normal(loc=0.0, scale=8.0)

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        if state == self.goal:
            reward = 50

        
        done = state == self.goal
        return state, reward, done, None

register(
    id='Puddle-v1',
    entry_point='PuddleDiscrete:PuddleD',
)
