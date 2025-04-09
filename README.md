# LunarLander-rl
A Deep Q-Network (DQN) with target network to solve the LunarLander-v2 environment using TensorFlow and Gymnasium. The implementation includes training, testing, replay buffer, epsilon decay, and target network updates for improved learning stability.

# 🚀 DQN LunarLander Agent

This project implements a **Deep Q-Network (DQN)** with a target network to train an agent for solving the `LunarLander-v2` environment from OpenAI Gymnasium. This environment is more challenging than CartPole and showcases advanced stability techniques in reinforcement learning.

## 📸 Demo

![Lunar Lander](https://media.giphy.com/media/WoD6JZnwap6s8/giphy.gif) <!-- Replace with your own demo/gif if you want -->

## 🚀 Features

- Deep Q-Network with experience replay
- Separate target network for stabilized learning
- Epsilon-greedy policy with decay
- Auto-saving of trained model
- Performance visualization
- Real-time testing with GUI

## 🛠️ Tech Stack

- **Language**: Python 3.11
- **Libraries**:
  - TensorFlow 2.15
  - Gymnasium (with Box2D)
  - NumPy
  - Matplotlib

## 📦 Installation

Create the environment using the provided YAML file:

```bash
conda env create -f environment.yml
conda activate rl-env
