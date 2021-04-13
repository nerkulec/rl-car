import gym
from gym import spaces
import pyglet
from pyglet import shapes
from pyglet.gl import glScalef, glTranslatef
import numpy as np
import math
from random import randrange as rand

# ignore
class Spec: # remove when env properly registered
  def __init__(self, id):
    self.id = id

tile_width = 40

class RLCar(gym.Env):
  metadata = {'render.modes': ['human', 'trajectories']}

  def __init__(self, file_name = '/home/bartek/rl-car/maps/map4.txt', num_rays = 12, draw_rays = True, n_trajectories = 10**4, step_cost = 0.1):
    super().__init__()
    self._max_episode_steps = 200

    self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    self.map = Map(file_name)
    self.car = Car()
    num_features = 2 # ile rodzajów terenu rozpoznajemy - aktuanie 2 - ściana i meta
    self.num_rays = num_rays
    self.observation_space = spaces.Box(
      low=np.concatenate([np.array([-1, -1], dtype=np.float32), np.zeros(num_features*self.num_rays, dtype=np.float32)]),
      high=np.concatenate([np.array([1, 1], dtype=np.float32), np.ones(num_features*self.num_rays, dtype=np.float32)]),
      dtype=np.float32, shape=(2+num_features*self.num_rays,)
    )
    self.step_cost = 0.1
    self.window = None
    self.spec = Spec('RLCar-v0')
    self.color = None
    self.i = 0
    self.n_trajectories = n_trajectories
    self.trajectories = [[] for _ in range(n_trajectories)]
    self.heatmap = []
    self.batch = None
    self.draw_rays = draw_rays
    self.opacity = None
    
  def get_obs(self):
    obs = [
      np.array([self.car.vel.x, self.car.vel.y]),
      self.map.get_closest(self.car.pos, '#', rays=self.num_rays, batch=self.draw_rays and self.batch or None),
      # self.map.get_closest(self.car.pos, 'O'), # wykrywanie oleju (nie robię narazie)
      # self.map.get_closest(self.car.pos, 'C'), # wykrywanie kota  (nie robię narazie)
      self.map.get_closest(self.car.pos, 'M', rays=self.num_rays)
    ]
    return np.concatenate(obs)

  def step(self, action):
    action = action*max_acc
    car = self.car
    if self.color is not None:
      prev_pos = Vec(car.pos.x, car.pos.y)
    car.update(action, self.ground)
    if self.color is not None and self.batch is not None:
      line = shapes.Line(prev_pos.x*tile_width, prev_pos.y*tile_width, car.pos.x*tile_width, car.pos.y*tile_width, color=self.color, batch=self.batch, width=self.color[0] == 255 and 2 or 1)
      line.opacity = self.opacity or 63
      self.trajectories[self.i%self.n_trajectories].append(line)
    self.ground = self.map[car.pos]

    self.steps += 1
    if self.steps >= 200:
      # time exceeded
      reward = -100 # or -150 or -50 or 0 or 50 ??
      done = True
      end_color = (102, 0, 204)
    elif self.ground == ' ' or self.ground == 'O':
      # nothing happens
      reward = -self.step_cost
      done = False
    elif self.ground == '#':
      # we hit a wall
      reward = -100
      done = True
      end_color = (255, 102, 0)
    elif self.ground == 'C':
      # we ran over a cat
      reward = -100
      done = False
    elif self.ground == 'M':
      # we reached the finish line
      reward = 1000
      done = True
      end_color = (0, 255, 0)
    else:
      raise Exception('Unsupported ground')

    if done:
      if self.color is not None and self.batch is not None:
        end = shapes.Circle(car.pos.x*tile_width, car.pos.y*tile_width, 4, color=end_color, batch=self.batch)
        end.opacity = 128
        self.trajectories[self.i%self.n_trajectories].append(end)
      self.i += 1
      if self.color is not None and self.batch is not None:
        self.trajectories[self.i%self.n_trajectories] = []

    if done and self.ground != 'M':
      # subtract distance from the finish line
      dist = math.sqrt((car.pos.x-self.map.M.x)**2+(car.pos.y-self.map.M.y)**2)
      reward -= dist*10

    obs = self.get_obs()
    return obs, reward/1000, done, {}
    
  def reset(self):
    self.car = Car()
    self.ground = ' '
    self.steps = 0
    return self.get_obs()
  
  def set_color(self, color):
    self.color = color    
    
  def render(self, mode='human', close=False):
    if mode == 'human':
      if self.window is None:
        self.window = pyglet.window.Window(self.map.width*tile_width, self.map.height*tile_width)
        glTranslatef(-1, 1, 0)
        glScalef(2/self.map.width/tile_width, -2/self.map.height/tile_width, 1)
        self.batch = pyglet.graphics.Batch()
        self.circle = shapes.Circle(self.car.pos.x*tile_width, self.car.pos.y*tile_width, tile_width/4, color=(255, 128, 128), batch=self.batch)
        self.map.draw(self.batch)
      self.circle.x = self.car.pos.x*tile_width
      self.circle.y = self.car.pos.y*tile_width
      self.window.clear()
      # Draw board
      self.batch.draw()
      # Draw car
      self.circle.draw()
      self.window.flip()
    if mode == 'trajectories':
      if self.window is None:
        self.window = pyglet.window.Window(self.map.width*tile_width, self.map.height*tile_width)
        glTranslatef(-1, 1, 0)
        glScalef(2/self.map.width/tile_width, -2/self.map.height/tile_width, 1)
        self.batch = pyglet.graphics.Batch()
        self.map.draw(self.batch)
      self.window.clear()
      self.batch.draw()
      self.window.flip()
    
  def close(self):
    super().close()
    if self.window is not None:
      self.window.close()
      self.window = None

oil_sigma = 0.2

max_vel = 0.15
max_acc = 0.02

class Car:
  def __init__(self, pos=None):
    if pos is None:
        self.pos = Vec(1.5, 1.5)
    else:
        self.pos = pos
    self.vel = Vec(0, 0)
  
  def update(self, acc, field):
    acc = Vec(acc)
    acc.limit(max_acc)
    if field == 'O':
        acc += Vec(np.random.normal(scale=oil_sigma, size=2))
    self.vel += acc
    self.vel.limit(max_vel)
    self.pos += self.vel

class Vec:
  def __init__(self, a, b=None):
    self.x, self.y = a if b is None else (a, b)
  
  def __add__(self, other):
    return Vec(self.x+other.x, self.y+other.y)
  
  def __iadd__(self, other):
    self.x+=other.x
    self.y+=other.y
    return self
  
  def __sub__(self, other):
    return Vec(self.x-other.x, self.y-other.y)
  
  def __neg__(self):
    return Vec(-self.x, -self.y)
  
  def mag(self):
    return math.sqrt(self.x**2 + self.y**2)
  
  def limit(self, r):
    m = self.mag()
    if m > r:
        self.x *= r/m
        self.y *= r/m
  
  def __eq__(self, other):
    return other is not None and self.x == other.x and self.y == other.y
  
  def copy(self):
    return Vec(self.x, self.y)
  
  def __repr__(self):
    return f"({self.x}, {self.y})"
  
  def __lt__(self, other):
    return self.x<other.x or (self.x==other.x and self.y < other.y)
  
  def __hash__(self):
    return hash((self.x, self.y))

class Map:
  def __init__(self, file_name = '~/rl-car/maps/map4.txt'):
    board = []
    with open(file_name, 'r') as f:
      for line in f:
        board.append([c for c in line[:-1]])
    self.width = len(board[0])
    self.height = len(board)
    self.board = board
    for y in range(self.height):
      for x in range(self.width):
        v = Vec(x, y)
        if self[v] == 'M':
          self.M = v + Vec(0.5, 0.5)

  def __getitem__(self, pos):
    try:
        return self.board[math.floor(pos.y)][math.floor(pos.x)]
    except:
        return '#'

  def __setitem__(self, pos, val):
    self.board[pos.y][pos.x] = val

  def __contains__(self, pos):
    return 0 <= pos.x < self.width and\
            0 <= pos.y < self.height and\
            self[pos] != '#'
              
  def get_closest(self, pos, field = '#', jump = 0.2, num_steps = 11, rays = 12, batch = None):
    if batch is not None:
        self.rays = []
    closest = np.zeros(rays)
    for i in range(rays):
      c = math.cos(i*2*math.pi/rays)
      s = math.sin(i*2*math.pi/rays)
      for j in range(1, num_steps+1):
        ray = pos + Vec(c*j*jump, s*j*jump)
        closest[i] = j
        if self[ray] == field:
          break
      if batch is not None:
        coef = math.floor(255*(num_steps-j)/(num_steps-1))
        self.rays.append(shapes.Line(
            pos.x*tile_width, pos.y*tile_width, ray.x*tile_width, ray.y*tile_width, 1,
            color=(coef, (255-coef)//2, 0), batch=batch))
    return (num_steps-closest)/(num_steps-1)

  def print(self):
    for y in range(self.height):
      for x in range(self.width):
        print(self.board[y][x], end='')
      print()
  
  def draw(self, batch):
    self.rects = []
    for y in range(self.height):
      for x in range(self.width):
        g = self[Vec(x, y)]
        if g == ' ':
          c = (0, 0, 0)
        elif g == '#':
          c = (128, 128, 128)
        elif g == 'C':
          c = (140, 120, 120)
        elif g == 'O':
          c = (40, 26, 13)
        elif g == 'M':
          c = (128, 128, 256)
        rect = shapes.Rectangle(x*tile_width, y*tile_width, tile_width, tile_width, color=c, batch=batch)
        self.rects.append(rect)
