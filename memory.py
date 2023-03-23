import torch


class ReplayBuffer:
    def __init__(self):
        self.buffer = []  # 缓冲区

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        l_s, l_a, l_r, l_s_, l_done = [], [], [], [], []
        for item in self.buffer:
            s, a, r, s_, done = item
            l_s.append(torch.tensor([s], dtype=torch.float))
            l_a.append(torch.tensor([a], dtype=torch.float))
            l_r.append(torch.tensor([r], dtype=torch.float))
            l_s_.append(torch.tensor([s_], dtype=torch.float))
            l_done.append(torch.tensor([done], dtype=torch.float))
        s = torch.cat(l_s, dim=0)
        r = torch.cat(l_r, dim=0)
        done = torch.cat(l_done, dim=0)
        a = torch.cat(l_a, dim=0)
        # r = torch.cat(l_r, dim=0)
        s_ = torch.cat(l_s_, dim=0)
        self.data = []
        return s, a, r, s_, done

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)
