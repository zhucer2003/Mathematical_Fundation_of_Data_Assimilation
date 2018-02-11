clear;set(0,'defaultaxesfontsize',20);format long
%%% p14.m Particle Filter (SIRS), sin map (Ex. 1.3)
%% setup

J=1e3;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=1;% observational noise variance is gamma^2
sigma=3e-1;% dynamics noise variance is sigma^2
C0=9e-2;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed
N=100;% number of ensemble members

m=zeros(J,1);v=m;y=m;c=m;U=zeros(J,N);% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=10*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator
U(1,:)=m(1)+sqrt(c(1))*randn(1,N);m(1)=sum(U(1,:))/N;% initial ensemble

%% solution % Assimilate!

for j=1:J  
    
    v(j+1)=alpha*sin(v(j)) + sigma*randn;% truth
    y(j)=H*v(j+1)+gamma*randn;% observation
 
    Uhat=alpha*sin(U(j,:))+sigma*randn(1,N);% ensemble predict
    d=y(j)-H*Uhat;% ensemble innovation  
    what=exp(-1/2*(1/gamma^2*d.^2));% weight update
    w=what/sum(what);% normalize predict weights   
 
    ws=cumsum(w);% resample: compute cdf of weights
    for n=1:N
        ix=find(ws>rand,1,'first');% resample: draw rand \sim U[0,1] and 
        % find the index of the particle corresponding to the first time 
        % the cdf of the weights exceeds rand.
        U(j+1,n)=Uhat(ix);% resample: reset the nth particle to the one 
        % with the given index above
    end
    
    m(j+1)=sum(U(j+1,:))/N;% estimator update
    c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))'/N;% covariance update
     
end

js=21;% plot truth, mean, standard deviation, observations
figure;plot([0:js-1],v(1:js));hold;plot([0:js-1],m(1:js),'m');
plot([0:js-1],m(1:js)+sqrt(c(1:js)),'r--');plot([1:js-1],y(1:js-1),'kx');
plot([0:js-1],m(1:js)-sqrt(c(1:js)),'r--');hold;grid;xlabel('iteration, j');
title('Particle Filter (Standard), Ex. 1.3');

figure;plot([0:J],(v-m).^2);hold;
plot([0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('iteration, j');title('Particle Filter (Standard) Error, Ex. 1.3')





