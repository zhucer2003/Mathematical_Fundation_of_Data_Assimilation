clear; set(0,'defaultaxesfontsize',20); format long
%%% p6c.m MCMC pCN algorithm algorithm in continuous time
%%% for double well (Ex 5.6) with noise
%% setup
m0=0;% prior initial condition mean
C0=1;% prior initial condition variance
gamma=0.5;% observational noise variance is gamma^2
tau=0.1; T=1;% timestep and final time of integration
epsilon=0.08; sigma=sqrt(2*epsilon);% dynamics noise variance is sigma^2
sd=1;rng(sd);% Choose random number seed 
t=[0:tau:T]; J=length(t);% number of points where we approximate u
%% truth
vt=zeros(1,J);  z=zeros(1,J); dz=zeros(1,J-1); dvt=zeros(1,J-1);% preallocate 
ut=[sqrt(C0)*randn,sigma*sqrt(tau)*randn(1,J-1)];% truth noise sequence
z(1)=0;dW_z=sqrt(tau)*randn(1,J-1);% truth noise sequence observation
vt(1)=ut(1);% truth initial condition 
Xi2=0; Xi1=0;
for j=1:J-1
    vt(j+1)=vt(j)+tau*vt(j)*(1-vt(j)^2)+ut(j+1); % create truth
    dvt(j)=vt(j+1)-vt(j);
    z(j+1)=z(j)+tau*vt(j)+gamma*dW_z(j); % create data
    dz(j)=z(j+1)-z(j);
    % calculate Xsi_2(v;z) from (5.27)
    Xi2=Xi2+(0.5*tau*vt(j)^2-vt(j)*dz(j))/gamma^2;
    Xi1=Xi1+(0.5*tau*(vt(j)*(1-vt(j)^2))^2- vt(j)*(1-vt(j)^2)*dvt(j))/sigma^2;
end
%% solution
% Markov Chain Monte Carlo: N forward steps of the
% Markov Chain on R^{J+1} with truth initial condition
N=1e5;% number of samples
V=zeros(N,J);% preallocate space to save time
v=vt;% truth initial condition 
beta=0.05;% step-size of pCN walker 
n=1; bb=0; rat(1)=0; 
m=[m0,zeros(1,J-1)];
while n<=N
    dW=[sqrt(C0)*randn,sigma*sqrt(tau)*randn(1,J-1)];% Brownian increments
    xi=cumsum(dW); % Brownian motion starting at a random initial condition
    w=m+sqrt(1-beta^2)*(v-m)+beta*xi;% propose sample from the pCN walker
    Xi2prop=0;  Xi1prop=0;
    for j=1:J-1
    Xi2prop=Xi2prop+(0.5*tau*w(j)^2-w(j)*dz(j))/gamma^2;
    Xi1prop=Xi1prop+0.5*tau*(w(j)*(1-w(j)^2))^2/sigma^2- ...
    w(j)*(1-w(j)^2)*(w(j+1)-w(j))/sigma^2;
    end
    if rand<exp(Xi2-Xi2prop+Xi1-Xi1prop)% accept or reject proposed sample
       v=w; Xi2=Xi2prop; Xi1=Xi1prop; bb=bb+1;% update the Markov chain
    end
   rat(n)=bb/n;% running rate of acceptance
   V(n,:)=v;% store the chain
   n=n+1;
end
% plot acceptance ratio and cumulative sample mean
figure;plot(rat);figure;plot(cumsum(V(1:N,end))./[1:N]');