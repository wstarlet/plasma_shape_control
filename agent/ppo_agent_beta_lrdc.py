import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
import wandb
from torch.distributions import Beta
from sklearn.metrics import explained_variance_score


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class ReplayBuffer:
    def __init__(self, args):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.done = []
        self.count = 0

    def clear_memory(self):
        del self.s[:]
        del self.a[:]
        del self.a_logprob[:]
        del self.r[:]
        del self.s_[:]
        del self.done[:]

class onnx_Actor(nn.Module):
    def __init__(self, args):
        super(onnx_Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.actor_hidden_width)
        self.fc2 = nn.Linear(args.actor_hidden_width, args.actor_hidden_width)
        self.fc3 = nn.Linear(args.actor_hidden_width, args.actor_hidden_width)
        self.alpha_layer = nn.Linear(args.actor_hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.actor_hidden_width, args.action_dim)
        self.activate_func = nn.ReLU() 

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        mean = alpha / (alpha + beta)
        return mean


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.actor_hidden_width)
        self.fc2 = nn.Linear(args.actor_hidden_width, args.actor_hidden_width)
        self.fc3 = nn.Linear(args.actor_hidden_width, args.actor_hidden_width)
        self.alpha_layer = nn.Linear(args.actor_hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.actor_hidden_width, args.action_dim)
        self.activate_func = nn.ReLU()

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        s = self.activate_func(self.fc3(s))

        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0

        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)
        return mean



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.critic_hidden_width)
        self.fc2 = nn.Linear(args.critic_hidden_width, args.critic_hidden_width)
        self.fc3 = nn.Linear(args.critic_hidden_width, 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.activate_func = nn.LeakyReLU()

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.dropout(s)
        s = self.activate_func(self.fc2(s))
        s = self.dropout(s)
        v_s = self.fc3(s)
        return v_s


class ppo_agent():
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.state_dim = args.state_dim

        self.delta_u_threshold = args.u_threshold  # single-step voltage change must not exceed the limit

        self.actor = Actor_Beta(args).to(self.device)
        self.onnx_actor = onnx_Actor(args)
        self.critic = Critic(args).to(self.device)

        self.set_adam_eps = args.set_adam_eps
        self.use_adv_norm = args.use_adv_norm
        self.use_lr_decay = args.use_lr_decay
        self.use_grad_clip = args.use_grad_clip

        if self.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s, a_last):
        a = self.actor.mean(s).detach()
        a_allow_up = a_last + self.delta_u_threshold
        a_allow_down = a_last - self.delta_u_threshold
        a_down = torch.clamp(a_allow_down, 0, 1)
        a_up = torch.clamp(a_allow_up, 0, 1)
        a = torch.clamp(a, a_down, a_up)
        return a

    def choose_action(self, s, a_last):
        with torch.no_grad():
            a_allow_up = a_last+self.delta_u_threshold
            a_allow_down = a_last-self.delta_u_threshold

            dist = self.actor.get_dist(s)
            dist_entropy = dist.entropy().sum(1, keepdim=True)
            a = dist.sample()  # Sample the action according to the probability distribution
            a_down = torch.clamp(a_allow_down, 0, 1)
            a_up = torch.clamp(a_allow_up, 0, 1)
            a = torch.clamp(a, a_down, a_up)
            a_logprob = dist.log_prob(a)  # The log probability density of the action
            # wandb.log({'action_entropy':dist_entropy})
        return a, a_logprob  # a.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s = torch.stack(replay_buffer.s).squeeze(1)  # Get training data [batch_size,1,6]->[batch_size,6]
        a = torch.stack(replay_buffer.a).squeeze(1)
        a_logprob = torch.stack(replay_buffer.a_logprob).squeeze(1)
        r = torch.stack(replay_buffer.r)
        s_ = torch.stack(replay_buffer.s_).squeeze(1)
        done = torch.tensor(replay_buffer.done, dtype=torch.float).unsqueeze(-1)
        """
            Calculate the advantage using GAE
            'done=True' represents the terminal of an episode(reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().flatten().numpy()), reversed(done.cpu().flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs

            y_true = v_target.squeeze(1).cpu().numpy()
            y_pred = vs.squeeze(1).cpu().numpy()
            exp_var = explained_variance_score(y_true,y_pred)
            wandb.log({'explained_variance': exp_var})
            print(f'explained_variance:{exp_var:.6f}')

            if self.use_adv_norm:  # advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)/17
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True).detach())

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                wandb.log({'actor_loss': actor_loss.mean().item()})

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5).to(self.device)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                wandb.log({'critic_loss': critic_loss.item()})

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5).to(self.device)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
            print(f'lr_a_now:{lr_a_now}')
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
            print(f'lr_c_now:{lr_c_now}')

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'actor_final.pt')
        torch.save(self.critic.state_dict(), path + 'critic_final.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor_final.pt', map_location=self.device))
        self.critic.load_state_dict(torch.load(path + 'critic_final.pt', map_location=self.device))

    def load_real_time(self, path):
        self.actor.load_state_dict(torch.load(path + 'actor_real_time.pt', map_location=self.device))
        self.critic.load_state_dict(torch.load(path + 'critic_real_time.pt', map_location=self.device))

    def save_real_time(self, path):
        torch.save(self.actor.state_dict(), path + 'actor_real_time.pt')
        torch.save(self.critic.state_dict(), path + 'critic_real_time.pt')

    def actor2onnx(self,path):
        actor_final = self.onnx_actor
        # final
        actor_final.load_state_dict(torch.load(path + 'actor_real_time.pt',map_location=self.device))
        for par in actor_final.parameters():
            print(par)

        inputs = torch.randn((self.state_dim,))
        torch.onnx.export(model=actor_final,
                          args=(inputs,),
                          f=path+'actor.onnx',
                          export_params=True,
                          opset_version=15,
                          input_names=['observation'],
                          output_names=["action"])








