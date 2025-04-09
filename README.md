# LunarLander-rl
A Deep Q-Network (DQN) with target network to solve the LunarLander-v2 environment using TensorFlow and Gymnasium. The implementation includes training, testing, replay buffer, epsilon decay, and target network updates for improved learning stability.

# 🚀 DQN LunarLander Agent

This project implements a **Deep Q-Network (DQN)** with a target network to train an agent for solving the `LunarLander-v2` environment from OpenAI Gymnasium. This environment is more challenging than CartPole and showcases advanced stability techniques in reinforcement learning.

## 📸 Visualisations

![image](https://github.com/user-attachments/assets/6266b4d4-faa1-4760-ad26-e98667db321a)

![image](https://github.com/user-attachments/assets/159f25e6-5c4a-4ce1-802b-39369b3829b5)


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
