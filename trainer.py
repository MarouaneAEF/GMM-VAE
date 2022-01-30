import torch 
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np 

class Trainer:
    def __init__(self,model, dataloader, optim, n_epoches):
        self.model = model
        self.train_dataloader, self.testdataloader = dataloader
        self.optim = optim
        self.n_epoches = n_epoches 

        self.optimizer = self.optim(self.model.parameters(), lr=1e-3)
        self._dev = torch.device("cuda")
        
    
    @staticmethod
    def loss_function(recon_X, X, mu_w, logvar_w, qz,mu_x, logvar_x, mu_px, logvar_px, x_sample, x_size, K):
        N = X.size(0) # batch size

        # 1. Reconstruction Cost = -E[log(P(y|x))]
        # for dataset like MNIST the recon_loss is binary cross entrpy 
        # recon_loss should be reconsidred depending on the nature of the dataset 
    
        recon_loss = F.binary_cross_entropy(recon_X, X,size_average=False)
        
        # recon_loss for normally distributed data, we can consider the negative log likelihood as a loss 
        # it is equivelent, statisticly speaking, to consider a binary corss entropy between real data distribution and the distribution that we are about 
        # to settle (model). That is : 
        #  E_x~Pdata [log Pdata(x) - log Pmodel(x)] = - E_x~Pdata [log Pmodel(x)] 
        # We can use negative LL, so sign is flipped
        # log(sigma) + 0.5*2*pi + 0.5*(x-mu)^2/sigma^2
        # recon_loss = 0.5 * torch.sum(logvar_recon_X + math.log(2*math.pi) + (X - mu_recon_X).pow(2)/logvar_recon_X.exp())

        # 2. KL( q(w) || p(w) )
        KLD_W = -0.5 * torch.sum(1 + logvar_w - mu_w.pow(2) - logvar_w.exp())

        # 3. KL( q(z) || p(z) )
        KLD_Z = torch.sum(qz * torch.log(K * qz + 1e-10))


        # 4. E_z_w[KL(q(x)|| p(x|z,w))]
        # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 )
        mu_x = mu_x.unsqueeze(-1)
        mu_x = mu_x.expand(-1, x_size, K)

        logvar_x = logvar_x.unsqueeze(-1)
        logvar_x = logvar_x.expand(-1, x_size, K)

        # shape (-1, x_size, K)
        KLD_QX_PX = 0.5 * (((logvar_px - logvar_x) + ((logvar_x.exp() + (mu_x - mu_px).pow(2))/logvar_px.exp())) - 1)

        # transpose to change dim to (-1, x_size, K)
        # KLD_QX_PX = KLD_QX_PX.transpose(1,2)
        qz = qz.unsqueeze(-1)
        qz = qz.expand(-1, K, 1)

        E_KLD_QX_PX = torch.sum(torch.bmm(KLD_QX_PX, qz))

        # 5. Entropy criterion

        # CV = H(Z|X, W) = E_q(x,w) [ E_p(z|x,w)[ - log P(z|x,w)] ]
        # compute likelihood

        x_sample = x_sample.unsqueeze(-1)
        x_sample =  x_sample.expand(-1, x_size, K)

        temp = 0.5 * x_size * np.log(2 * np.pi)
        # log likelihood
        llh = -0.5 * torch.sum(((x_sample - mu_px).pow(2))/logvar_px.exp(), dim=1) - 0.5 * torch.sum(logvar_px, dim=1) - temp

        lh = F.softmax(llh, dim=1)

        # entropy
        CV = torch.sum(torch.mul(torch.log(lh+1e-10), lh))

        loss = recon_loss + KLD_W + KLD_Z + E_KLD_QX_PX

        return loss, recon_loss, KLD_W, KLD_Z, E_KLD_QX_PX, CV


    def train(self, epoch):
        self.model.to(self._dev)
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            
            self.optimizer.zero_grad()

            data = data.to(self._dev)
            target = target.to(self._dev)

            mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
                x_sample = self.model(data)

            loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
                = self.loss_function(Y, data, mu_w, logvar_w, qz,
                mu_x, logvar_x, mu_px, logvar_px, x_sample, self.model.x_size, self.model.K)

            loss.backward()

            self.optimizer.step()

            
            
            if batch_idx%10 == 0: 
                status = f"Train {epoch}: [{batch_idx+1}/{len(self.train_dataloader)} loss: {loss.item():>7f} RecLoss: {BCE.item():>7f} E[KLD(QX||PX)]: {E_KLD_QX_PX.item():>7f} CV: {CV.item():>7f} KLD_W: {KLD_W.item():>7f} KLD_Z: {KLD_Z.item():>7f}]"
                print(status)


    def test(self, epoch):
        self.model.to(self._dev)
        self.model.eval()

        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.testdataloader):
                data = data.to(self._dev)
                
                target = target.to(self._dev)

                mu_x, logvar_x, mu_px, logvar_px, qz, Y , mu_w, logvar_w, \
                    x_sample = self.model(data)

                
                loss, BCE, KLD_W, KLD_Z, E_KLD_QX_PX, CV \
                = self.loss_function(Y, data, mu_w, logvar_w, qz,
                mu_x, logvar_x, mu_px, logvar_px, x_sample, self.model.x_size, self.model.K)

                if batch_idx%10 == 0: 
                    status = f"Train {epoch}: [{batch_idx+1}/{len(self.train_dataloader)} loss: {loss.item():>7f} RecLoss: {BCE.item():>7f} E[KLD(QX||PX)]: {E_KLD_QX_PX.item():>7f} CV: {CV.item():>7f} KLD_W: {KLD_W.item():>7f} KLD_Z: {KLD_Z.item():>7f}]"
                    print(status)

                
                n = min(data.size(0), 32)
                
                comparision = torch.cat([data[:n], Y[:n]])
                
                
                save_image(comparision.cpu(),
                        './results/reconstruction_'+str(epoch)+'.png', nrow=16)

        
