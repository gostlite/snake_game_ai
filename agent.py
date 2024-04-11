import torch
import numpy as np
import random
from collections import deque
from game import SnakeGameAi, Direction, Point
from model import Linear_QNet, Qtrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE =1000
learning_rate = 0.001

class Agent:
    def __init__(self) -> None:
        self.number_of_games =0
        self.epislon = 0 #randomness
        self.gamma = 0.9 #discount_rate
        self.memory = deque(maxlen=MAX_MEMORY) #popleft()
        self.model = Linear_QNet(11,256,3)
        self.trainer = Qtrainer(self.model,lr=learning_rate, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l=Point(head.x -20, head.y)
        point_r=Point(head.x +20, head.y)
        point_u=Point(head.x, head.y-20)
        point_d=Point(head.x , head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state=[
            # danger straight
            (dir_r and game.is_collision(point_r))or
            (dir_l and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_l and game.is_collision(point_u))or
            (dir_r and game.is_collision(point_d)),

            #Danger left
            (dir_d and game.is_collision(point_r))or
            (dir_u and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            #move direction
            dir_r,
            dir_l,
            dir_u,
            dir_d,

            #food location 
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x ,# food right
            game.food.y < game.head.y, #food down
            game.food.y > game.head.y, # food up


        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state,done):
        self.memory.append((state, action, reward, next_state,done)) # popleft if max_memory


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuple
        else:
            mini_sample = self.memory
            states,actions, rewards,next_states,dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states,dones)

    def train_short_memory(self,state, action, reward, next_state,done):
        self.trainer.train_step(state, action, reward, next_state,done)

    def get_action(self, state):
        # random moves: tradeoff exploration / explotation
        self.epislon = 80 - self.number_of_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epislon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # convert the max value to one
            final_move[move] = 1
        return final_move

def train():
        plot_score = []
        plot_mean_score = []
        total_score =0
        record = 0
        agent = Agent()
        game = SnakeGameAi()
        while True:
            # old state
            old_state = agent.get_state(game)

            # get move
            final_move = agent.get_action(old_state)

            #perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            #train short memory
            agent.train_short_memory(old_state,final_move,reward,state_new,done)

            # remember
            agent.remember(old_state,final_move,reward,state_new,done)

            if done:
                #train on long memory, plot result
                game.reset()
                agent.number_of_games+=1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                print(f"Game: {agent.number_of_games}", f"Score: {score}", f"Record: {record}")
                plot_score.append(score)
                total_score +=score
                mean_score = total_score /agent.number_of_games
                plot_mean_score.append(mean_score)
                plot(plot_score, plot_mean_score)




if __name__ == "__main__":
    train()