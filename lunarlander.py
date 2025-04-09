import gymnasium as gym
import numpy as np
import random
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            
            self.model.fit(state, target, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_lunar_lander(episodes=1000):
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    batch_size = 64
    update_target_every = 5  # Update target network every 5 episodes
    
    scores = []
    recent_scores = deque(maxlen=100)

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for _ in range(1000):  # Max steps per episode
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        # Train the agent with experiences in memory
        if len(agent.memory) > batch_size:
            for _ in range(10):  # Multiple training steps per episode
                agent.replay(batch_size)
        
        # Update target network
        if e % update_target_every == 0:
            agent.update_target_model()
        
        scores.append(total_reward)
        recent_scores.append(total_reward)
        avg_score = np.mean(recent_scores)
        
        print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # If average score over last 100 episodes is over 200, consider it solved
        if len(recent_scores) == 100 and np.mean(recent_scores) >= 200:
            print(f"Environment solved in {e+1} episodes!")
            break
    
    # Save trained model
    agent.model.save("lunar_lander_dqn.h5")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Lunar Lander Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()
    
    return agent

def test_lunar_lander(agent=None, episodes=10, delay=0.025):
    env = gym.make('LunarLander-v2', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    if agent is None:
        agent = DQNAgent(state_size, action_size)
        agent.model = tf.keras.models.load_model("lunar_lander_dqn.h5")
        # Turn off exploration for testing
        agent.epsilon = 0
    
    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        
        for _ in range(1000):
            action = agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}")
                break

            #time.sleep(delay)

agent = train_lunar_lander(episodes=20)
test_lunar_lander(agent)