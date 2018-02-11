clear; set(0,'defaultaxesfontsize',20); format long
%%% p5c.m MCMC INDEPENDENCE DYNAMICS SAMPLER algorithm in continuous time
%%% for double well (Ex 5.6) with noise

%% setup
m0=0;% prior initial condition mean
C0=1;% prior initial condition variance
gamma=0.5; % observational noise variance is gamma^2
tau=0.1; T=1;% timestep and final time of integraion
epsilon=0.08; sigma=sqrt(2*epsilon);% dynamics noise variance is sigma^2
sd=1;rng(sd);% Choose random number seed 
t=[0:tau:T]; J=length(t);% number of points where we approximate u
 
%% truth
vt=zeros(J,1);  z=zeros(J,1); dz=zeros(1,J-1);% preallocate space to save time
ut=sqrt(C0)*randn; z(1)=0;
dW_v=sqrt(tau)*randn(J,1);% truth noise sequence model
dW_z=sqrt(tau)*randn(J,1);% truth noise sequence observation 
vt(1)=ut(1);% truth initial condition 
Xi2=0; 

for j=1:J-1
    vt(j+1)=vt(j)+tau*vt(j)*(1-vt(j)^2)+sigma*dW_v(j);% create truth
    z(j+1)=z(j)+tau*vt(j)+gamma*dW_z(j);% create data
    dz(j)=z(j+1)-z(j);
    % calculate Xsi_2(v;z) from (5.27)
    Xi2=Xi2+(0.5*tau*vt(j)^2-vt(j)*dz(j))/gamma^2;
end

%% solution
% Markov Chain Monte Carlo: N forward steps of the
% Markov Chain on R^{J+1} with truth initial condition
N=1e4;% number of samples
V=zeros(N,J);% preallocate space to save time
v=vt;% truth initial condition 
n=1; bb=0; rat(1)=0; 
while n<=N
    w(1)=sqrt(C0)*randn;% propose sample from the prior distribution
    Xi2prop=0; 
    for j=1:J-1
    w(j+1)=w(j)+tau*w(j)*(1-w(j)^2)+sigma*sqrt(tau)*randn;
    Xi2prop=Xi2prop+(0.5*tau*w(j)^2-w(j)*dz(j))/gamma^2;
    end
    if rand<exp(Xi2-Xi2prop)% accept or reject proposed sample
       v=w; Xi2=Xi2prop; bb=bb+1;% update the Markov chain
    end
   rat(n)=bb/n;% running rate of acceptance
   V(n,:)=v;% store the chain
   n=n+1;
end
% plot acceptance ratio and cumulative sample mean
figure;plot(rat);figure;plot(cumsum(V(1:N,end))./[1:N]')
xlabel('samples N');ylabel('(1/N) \Sigma_{n=1}^N v_0^{(n)}')
figure; plot([1:1:N],V(:,end))
