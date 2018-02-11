clear;set(0,'defaultaxesfontsize',20);format long
%%% p11.m  Extended Kalman Filter, sin map (Ex. 1.3)
%% setup

J=1e3;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=1;% observational noise variance is gamma^2
sigma=3e-1;% dynamics noise variance is sigma^2
C0=9e-2;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed

m=zeros(J,1);v=m;y=m;c=m;% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=10*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator

%% solution % assimilate!

for j=1:J 
    
    v(j+1)=alpha*sin(v(j)) + sigma*randn;% truth
    y(j)=H*v(j+1)+gamma*randn;% observation

    mhat=alpha*sin(m(j));% estimator predict
    chat=alpha*cos(m(j))*c(j)*alpha*cos(m(j))+sigma^2;% covariance predict  
    
    d=y(j)-H*mhat;% innovation
    K=(chat*H')/(H*chat*H'+gamma^2);% Kalman gain
    m(j+1)=mhat+K*d;% estimator update
    c(j+1)=(1-K*H)*chat;% covariance update
    
end

js=21;% plot truth, mean, standard deviation, observations
figure;plot([0:js-1],v(1:js));hold;plot([0:js-1],m(1:js),'m');
plot([0:js-1],m(1:js)+sqrt(c(1:js)),'r--');plot([1:js-1],y(1:js-1),'kx');
plot([0:js-1],m(1:js)-sqrt(c(1:js)),'r--');hold;grid;xlabel('iteration, j');
title('ExKF, Ex. 1.3')

figure;plot([0:J],c);hold
plot([0:J],cumsum(c)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('iteration, j');
title('ExKF Covariance, Ex. 1.3');

figure;plot([0:J],(v-m).^2);hold;
plot([0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('iteration, j');
title('ExKF Error, Ex. 1.3')









