clear; set(0,'defaultaxesfontsize',20); format long
%%% p4c.m MCMC RWM algorithm for double well (Ex. 5.6)
%% setup

C0=5;% variance of the prior
m0=4;% mean of the prior
sd=1;rng(sd);% choose random number seed
T=1; tau=0.1; t=[0:tau:T]; N=length(t);% time discretization
gamma=1;% observational noise variance is gamma^2

%% truth
vt(1)=0.5;  z(1)=0;% truth initial condition
dW=sqrt(tau)*randn(N-1,1);% precalculate the Brownian increments used
Jdet=1/2/C0*(vt(1)-m0)^2;% background penalization
Xi3=0;% initialization model-data misfit functional
    for i=1:N-1
        % can be replaced Psi for each problem
        vt(i+1)=vt(i)+tau*(vt(i)-vt(i)^3);
        z(i+1)=z(i)+tau*vt(i)+gamma*dW(i); % create data
        Xi3=Xi3+(tau*0.5*vt(i)^2-vt(i)*(z(i+1)-z(i)))/gamma^2;
    end
Idet=Jdet+Xi3;% compute log posterior of the truth

%% solution
% Markov Chain Monte Carlo: N forward steps of the
% Markov Chain on R (with truth initial condition)
M=1e5;% number of samples
V=zeros(M,1);% preallocate space to save time
beta=0.5;% step-size of random walker 
v=vt(1);% truth initial condition (or else update I0) 
n=1; bb=0; rat(1)=0; 
while n<=M
    w=v+sqrt(2*beta)*randn;% propose sample from random walker
    vv(1)=w;
    Jdetprop=1/2/C0*(w-m0)^2;% background penalization
    Xi3prop=0;	
    for i=1:N-1
            vv(i+1)=vv(i)+tau*(vv(i)-vv(i)^3);
            Xi3prop=Xi3prop+(tau*0.5*vv(i)^2-vv(i)*(z(i+1)-z(i)))/gamma^2;
    end
    Idetprop=Jdetprop+Xi3prop;% compute log posterior of the proposal 	

    if rand<exp(Idet-Idetprop)% accept or reject proposed sample
        v=w; Idet=Idetprop; bb=bb+1;% update the Markov chain
    end
    rat(n)=bb/n;% running rate of acceptance
    V(n)=v;% store the chain
    n=n+1 ;
end
dx=0.05; v0=[-10:dx:10]; Z=hist(V,v0);% construct the posterior histogram 
figure(1), plot(v0,Z/trapz(v0,Z),'k','Linewidth',2)% visualize the posterior
