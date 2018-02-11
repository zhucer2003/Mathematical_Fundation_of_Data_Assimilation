clear;set(0,'defaultaxesfontsize',20);format long
%%% p11c.m  Extended Kalman-Bucy Filter, double-well
%% setup

J=1e4;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=.1;% observational noise variance is gamma^2
sigma=3e-1;% dynamics noise variance is sigma^2
C0=1;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed

m=zeros(J,1);v=m;z=m;z(1)=0;c=m;% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=2*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator
tau=0.01;% time discretization is tau

%% solution % assimilate!

for j=1:J 
    
    v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) + sigma*sqrt(tau)*randn;% truth
    z(j+1)=z(j)+tau*H*v(j+1) + gamma*sqrt(tau)*randn;% observation

    mhat=m(j)+tau*alpha*(m(j)-m(j)^3);% estimator predict
    chat=(1+tau*alpha*(1-3*m(j)^2))*c(j)* ...
        (1+tau*alpha*(1-3*m(j)^2))+sigma^2*tau;% covariance predict  
    
    d=(z(j+1)-z(j))/tau-H*mhat;% innovation
    K=(tau*chat*H')/(H*chat*H'*tau+gamma^2);% Kalman gain
    m(j+1)=mhat+K*d;% estimator update
    c(j+1)=(1-K*H)*chat;% covariance update
    
end

js=201;% plot truth, mean, standard deviation, observations
figure;plot(tau*[0:js-1],v(1:js));hold;plot(tau*[0:js-1],m(1:js),'m');
plot(tau*[0:js-1],m(1:js)+sqrt(c(1:js)),'r--');
plot(tau*[0:js-1],m(1:js)-sqrt(c(1:js)),'r--');hold;grid;xlabel('t');
title('ExKF')

figure;plot(tau*[0:J],c);hold
plot(tau*[0:J],cumsum(c)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('t');
title('ExKF Covariance');

figure;plot(tau*[0:J],(v-m).^2);hold;
plot(tau*[0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('t');
title('ExKF Error')









