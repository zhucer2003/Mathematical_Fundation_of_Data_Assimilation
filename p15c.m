clear;set(0,'defaultaxesfontsize',20);format long
%%% p15c.m Particle Filter (SIRS, OP), double-well
%% setup
J=1e4;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=.1;% observational noise variance is gamma^2
sigma=3e-1;% dynamics noise variance is sigma^2
m0=0; C0=1;% prior initial condition mean and variance
sd=1;rng(sd);% Choose random number seed
N=10;% number of ensemble members

m=zeros(J,1);v=m;z=m;z(1)=0;c=m;U=zeros(J,N);% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=2*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator
U(1,:)=m(1)+sqrt(c(1))*randn(1,N);m(1)=sum(U(1,:))/N;% initial ensemble
tau=0.01;st=sigma*sqrt(tau);% time discretization is tau

%% solution % Assimilate!
for j=1:J   
    v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) + st*randn;% truth
    z(j+1)=z(j)+tau*H*v(j+1) + gamma*sqrt(tau)*randn;% observation
      
    Sig=inv(inv(sigma^2*tau)+H'*inv(gamma^2/tau)*H);% optimal proposal cov
    em=Sig*(inv(sigma^2*tau)*(U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3))+ ...
        H'*inv(gamma^2/tau)*(z(j+1)-z(j))/tau);% optimal proposal mean
    Uhat=em+sqrt(Sig)*randn(1,N);% ensemble optimally importance sampled     
    
    d=(z(j+1)-z(j))/tau-H*(U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3));% ensemble innov  
    what=exp(-1/2/(sigma^2*tau+gamma^2/tau)*d.^2);% weight update
    w=what/sum(what);randn(1,N);% normalize predict weights 
    
    ws=cumsum(w);% resample: compute cdf of weights
    for n=1:N
        ix=find(ws>rand,1,'first');% resample (i)
        U(j+1,n)=Uhat(ix);% resample (ii)
    end
    
    m(j+1)=sum(U(j+1,:))/N;% estimator update
    c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))'/N;% covariance update
end

js=201;% plot truth, mean, standard deviation, observations
figure(1);plot(tau*[0:js-1],v(1:js));hold;plot(tau*[0:js-1],m(1:js),'m');
plot(tau*[0:js-1],m(1:js)+sqrt(c(1:js)),'r--');
plot(tau*[0:js-1],m(1:js)-sqrt(c(1:js)),'r--');hold;grid;xlabel('t');
title('Particle Filter (Optimal)');
figure(2);plot(tau*[0:J],(v-m).^2);hold;
plot(tau*[0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('t');title('Particle Filter (Optimal) Error')
figure(3);plot(tau*[0:J],(v-m).^2);hold;
plot(tau*[0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('t');
title('Particle Filter (Optimal) Error')







