#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 15:17:42
LastEditor: John
LastEditTime: 2021-09-26 22:02:00
Discription:
Environment:
'''
import os
import numpy as np
import torch
import torch.optim as optim
from model import Actor, Critic
from memory import ReplayBuffer
import torch.nn.functional as F

K_epoch = 8
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2


class PPO(object):
    def __init__(self, state_dim, action_dim, cfg):
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, cfg.hidden_dim).to(self.device)
        self.old_actor = Actor(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.old_critic = Critic(state_dim, cfg.hidden_dim).to(self.device)
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        # self.memory = PPOMemory(cfg.batch_size)
        self.memory = ReplayBuffer()
        self.loss = 0

    def choose_action(self, observation):
        with torch.no_grad():
            observation = observation.to(self.device)
            mu, sigma = self.old_actor(observation)
            dis = torch.distributions.normal.Normal(mu, sigma)
            a = dis.sample()
        return a

    def update(self):
        # self.step += 1
        s, a, r, s_, done = self.memory.sample()
        s = s.to(self.device).squeeze()
        a = a.to(self.device).squeeze()
        a = a.unsqueeze(1)
        r = r.to(self.device).squeeze()
        r = r.unsqueeze(1)
        s_ = s_.to(self.device).squeeze()
        done = done.to(self.device).squeeze()
        done = done.unsqueeze(1)
        for _ in range(K_epoch):
            with torch.no_grad():
                '''loss_v'''
                temp = self.old_critic(s_)
                td_target = r + GAMMA * self.old_critic(s_) * (1 - done)
                '''loss_pi'''
                mu, sigma = self.old_actor(s)
                old_dis = torch.distributions.normal.Normal(mu, sigma)
                log_prob_old = old_dis.log_prob(a)
                td_error = r + GAMMA * self.critic(s_) * (1 - done) - self.critic(s)
                td_error = td_error.detach().cpu().numpy()
                A = []
                adv = 0.0
                for td in td_error[::-1]:
                    adv = adv * GAMMA * LAMBDA + td[0]
                    A.append(adv)
                A.reverse()
                A = torch.tensor(A, dtype=torch.float).reshape(-1, 1)
                A = A.to(self.device)

            mu, sigma = self.actor(s)
            new_dis = torch.distributions.normal.Normal(mu, sigma)
            log_prob_new = new_dis.log_prob(a)
            ratio = torch.exp(log_prob_new - log_prob_old)
            L1 = ratio * A
            L2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * A
            loss_pi = -torch.min(L1, L2).mean()
            self.actor.optim.zero_grad()
            loss_pi.backward()
            self.actor.optim.step()

            loss_v = F.mse_loss(td_target.detach(), self.critic(s))

            self.critic.optim.zero_grad()
            loss_v.backward()
            self.critic.optim.step()
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

    # def save(self, path):
    #     actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
    #     critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
    #     torch.save(self.actor.state_dict(), actor_checkpoint)
    #     torch.save(self.critic.state_dict(), critic_checkpoint)
    #
    # def load(self, path):
    #     actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
    #     critic_checkpoint = os.path.join(path, 'ppo_critic.pt')
    #     self.actor.load_state_dict(torch.load(actor_checkpoint))
    #     self.critic.load_state_dict(torch.load(critic_checkpoint))

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))
