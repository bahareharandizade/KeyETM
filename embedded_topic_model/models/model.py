import torch
import torch.nn.functional as F
from torch import nn


class Model(nn.Module):
    def __init__(
            self,
            device,
            theta_act,
            gamma_prior,
            gamma_prior_bin,
            lambda_theta,
            lambda_alpha,
            num_topics,
            vocab_size,
            t_hidden_size,
            rho_size,
            emsize,
            embeddings=None,
            train_embeddings=True,
            enc_drop=0.5,
            debug_mode=False):
        super(Model, self).__init__()

        # define hyperparameters
        self.theta_act= theta_act
        self.gamma_prior_bin = gamma_prior_bin
        self.gamma_prior = gamma_prior
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.debug_mode = debug_mode
        self.theta_act = self.get_activation(theta_act)
        self.lambda_theta = lambda_theta
        self.lambda_alpha = lambda_alpha
        self.device = device
        self.variance = 0.995
        self.num_topics = num_topics

        # define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            self.rho = embeddings.clone().float().to(self.device)

        # define the matrix containing the topic embeddings
        # nn.Parameter(torch.randn(rho_size, num_topics))
        self.prior_mean   = torch.Tensor(1, num_topics).fill_(0)
        self.prior_var    = torch.Tensor(1, num_topics).fill_(self.variance)
        self.prior_mean   = nn.Parameter(self.prior_mean, requires_grad=False)
        self.prior_var    = nn.Parameter(self.prior_var, requires_grad=False)
        self.prior_logvar = nn.Parameter(self.prior_var.log(), requires_grad=False)
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
        #self.alphas = nn.Sequential(
        #     nn.Linear(rho_size, 300,bias=False),
        #     self.theta_act,
        #     nn.Linear(300, rho_size,bias=False),
        #     self.theta_act,
        #     nn.Linear(rho_size,num_topics,bias=False),

        #)
        # define variational distribution for \theta_{1:D} via amortizartion
        #self.q_theta = nn.Sequential(
        #    nn.Linear(vocab_size, t_hidden_size),
        #    self.theta_act,
        #    nn.Linear(t_hidden_size, t_hidden_size),
        #    self.theta_act,
        #)
        self.en1_fc = nn.Linear(vocab_size, t_hidden_size)
        self.en1_ac = self.theta_act
        self.en2_fc = nn.Linear(t_hidden_size, t_hidden_size)
        self.en2_ac = self.theta_act
        
               
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.mean_bn = nn.BatchNorm1d(num_topics)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logvar_bn = nn.BatchNorm1d(num_topics)
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            act = nn.Tanh()
            if self.debug_mode:
                print('Defaulting to tanh activation')
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            #print(mu)
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        encoded1 = self.en1_fc(bows)
        encoded1_ac = self.en1_ac(encoded1)
        encoded2 = self.en2_fc(encoded1_ac)
        encoded2_ac = self.en2_ac(encoded2)
        #q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            #q_theta = self.t_drop(q_theta)
             q_theta =  self.t_drop(encoded2_ac)
        else:
             q_theta = encoded2_ac
        mu_theta = self.mu_q_theta(q_theta)
        #posterior_mean = self.mean_bn(mu_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        #posterior_logvar = self.logvar_bn(logsigma_theta)
        #kl_theta = -0.5 * \
        #    torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta
    
    def get_gamma(self):
        #weights = []
        #for name, param in self.q_theta.named_parameters():
           
               #weights.append(param.data)
        
        #q_theta_w0 = weights[0]
        #q_theta_w1 = weights[2]
        q_theta_w0=self.en1_fc.weight
        q_theta_b0= self.en1_fc.bias
        q_theta_w1 = self.en2_fc.weight
        q_theta_b1 = self.en2_fc.bias
        mean_w = self.mu_q_theta.weight
        mean_b = self.mu_q_theta.bias
        #print(mean_w.shape) 
        #print("***************************")
        w1 = F.softplus(q_theta_w0.t()+q_theta_b0)
        #print(w1.shape)
        w2 = F.softplus(F.linear(w1,q_theta_w1,q_theta_b1))
        #print(w2.shape)
        wdr = F.dropout(w2, self.enc_drop)
        #print(wdr.shape)
        wo_mean = F.softmax(F.linear(wdr, mean_w,mean_b), dim=-1)
        return wo_mean
        
        
    def get_beta(self):
        try:
            # torch.mm(self.rho, self.alphas)
            logit = self.alphas(self.rho.weight)
        except BaseException:
            logit = self.alphas(self.rho)
        beta = F.softmax(
            logit, dim=0).transpose(
            1, 0)  # softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, mu_theta, logsigma_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        # get \theta
        if theta is None:
            theta, posterior_mean, posterior_logvar = self.get_theta(normalized_bows)
        else:
            kld_theta = None
        #print(normalized_bows.shape)
        #print(theta)
        # get \beta
        beta = self.get_beta()
              
        # get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)


        posterior_var    = posterior_logvar.exp()
        prior_mean   = self.prior_mean.expand_as(posterior_mean)
        prior_var    = self.prior_var.expand_as(posterior_mean)
        prior_logvar = self.prior_logvar.expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kld_theta = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)
        kld_theta = kld_theta.mean()
        #print(bows)
        #get gamma loss
        #lambda_c = self.ld
        N, _ = bows.size()
        gamma_mean = self.get_gamma()
        #print(gamma_mean[95:100])
        #print(self.gamma_prior.shape)
        x_boolean = (bows > 0).unsqueeze(dim=-1)
        
        x_gamma_boolean = ((self.gamma_prior_bin[:N, :, :].expand(N, -1, -1) > 0) & x_boolean)
        #print(x_gamma_boolean)
        beta_boolean = (self.gamma_prior[:, :] > 0)
        
        beta_boolean = beta_boolean.permute(1,0)        
        #print(beta_boolean.shape)
        #print(beta.shape)
        gp = self.gamma_prior.expand(N, -1, -1)
        GL1 = self.lambda_theta * ((gp - (x_gamma_boolean.int()*gamma_mean))**2).sum((1, 2))
        
        #print(gamma_mean.shape)
        #print(x_gamma_boolean.shape)
        GL2 = self.lambda_alpha * ((self.gamma_prior.T-(beta_boolean.int()*beta))**2).sum(1)
        
        
        GL1 = GL1.mean() 
        GL2 = GL2.mean()
                
        if aggregate:
            recon_loss =  recon_loss.mean()
            #recon_loss = 0.8 * recon_loss.mean()
        return recon_loss, kld_theta,GL1,GL2
