import torch



class TLWLoss():
    def __init__(self, WeightNet, Discrimiator, w_optim, base='l1'):
        self.WNet = WeightNet
        self.D = Discrimiator
        self.WNetOptimizer = w_optim
        self.base = base

    def updateWeightNet(self, est, ref):
        self.WNetOptimizer.zero_grad()
        if (self.base == 'l1'):
            w = self.WNet(torch.cat([ref, est], dim=1))
            combined1 = (1 - w) * est.detach() + w * ref  # + noise
            combined2 = w * est.detach() + (1 - w) * ref  # + noise
        elif (self.base == 'l2'):
            w = self.WNet(torch.cat([ref, est], dim=1))
            combined1 = (1 - w) * est.detach() + w * ref  # + noise
            combined2 = w * est.detach() + (1 - w) * ref  # + noise

        w_loss = torch.mean(
            torch.log(self.D(combined1, ref, normalize=False)) - torch.log(self.D(combined2, ref, normalize=False)))

        w_loss.backward()
        self.WNetOptimizer.step()

        return w_loss.item(), torch.mean(w).item()

    def forward(self, est, ref):
        if (self.base == 'l1'):
            l = torch.abs(est - ref)
            w = self.WNet(torch.cat([ref, est], dim=1))
            loss = torch.mean(l * w.detach())  # /torch.mean(w).item()
        elif (self.base == 'l2'):
            l = (est - ref) ** 2
            w = self.WNet(torch.cat([ref, est], dim=1))
            loss = torch.mean(l * w.detach() / (torch.abs(est - ref).detach() + 0.1))  # /torch.mean(w).item()

        return loss


class TLWLossStochastic():
    def __init__(self, WeightNetStochastic, Discrimiator, w_optim, base='l1',device = 'cuda'):
        self.WNetStochastic = WeightNetStochastic
        self.D = Discrimiator
        self.WNetOptimizer = w_optim
        self.base = base
        self.device = device


    def updateWeightNet(self, est, ref):
        self.WNetOptimizer.zero_grad()
        if (self.base == 'l1'):
            w = self.WNetStochastic(torch.cat([ref, est], dim=1))
            combined1 = (1 - w) * est.detach() + w * ref  # + noise
            combined2 = w * est.detach() + (1 - w) * ref  # + noise
        elif (self.base == 'l2'):
            w = self.WNetStochastic(torch.cat([ref, est], dim=1))
            combined1 = (1 - w) * est.detach() + w * ref  # + noise
            combined2 = w * est.detach() + (1 - w) * ref  # + noise

        w_loss = torch.mean(
            torch.log(torch.abs(self.D(combined1, ref, normalize=False))+0.001) - torch.log(torch.abs(self.D(combined2, ref, normalize=False))+0.001))

        w_loss.backward()
        self.WNetOptimizer.step()

        return w_loss.item(), torch.mean(w).item()

    def forward(self, est, ref):
        if (self.base == 'l1'):
            l = torch.abs(est - ref)
            sum_p_w = torch.zeros(est.size(0), device=self.device)
            loss = torch.zeros(est.size(0), device=self.device)

            with torch.no_grad():
                ws = self.WNetStochastic(torch.cat([ref, est], dim=1), 3)
            for w in ws:
                with torch.no_grad():
                    combined1 = (1 - w) * est.detach() + w * ref  # + noise
                    combined2 = w * est.detach() + (1 - w) * ref  # + noise
                    p_w = (self.D(combined2, ref, normalize=False) / (torch.abs(self.D(combined1, ref,
                                                                       normalize=False)) + 0.001)).detach().squeeze()
                ex = torch.mean(torch.mean(torch.mean(l * w.detach(), dim=1), dim=1), dim=1)
                sum_p_w += p_w
                loss += p_w * torch.exp(-ex)

            loss = torch.mean(- torch.log(loss / sum_p_w))

        elif (self.base == 'l2'):
            l = (est - ref) ** 2
            sum_p_w = torch.zeros(est.size(0), device=self.device)
            loss = torch.zeros(est.size(0), device=self.device)
            with torch.no_grad():
                ws = self.WNetStochastic(torch.cat([ref, est], dim=1), 3)

            for w in ws:
                with torch.no_grad():
                    combined1 = (1 - w) * est.detach() + w * ref  # + noise
                    combined2 = w * est.detach() + (1 - w) * ref  # + noise
                    p_w = (self.D(combined2, ref, normalize=False) / (torch.abs(self.D(combined1, ref,
                                                                       normalize=False)) + 0.001)).detach().squeeze()
                ex = torch.mean(torch.mean(torch.mean(l * w.detach() / (torch.abs(est - ref).detach() + 0.2),
                                                      dim=1), dim=1), dim=1)
                sum_p_w += p_w
                loss += p_w * torch.exp(-ex)
            loss = torch.mean(-torch.log(loss / sum_p_w))

        return loss
