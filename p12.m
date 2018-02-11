clear;set(0,'defaultaxesfontsize',20);format long
%%% p12.m Ensemble Kalman Filter (PO), sin map (Ex. 1.3)
%% setup

J=1e5;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=1;% observational noise variance is gamma^2
sigma=3e-1;% dynamics noise variance is sigma^2
C0=9e-2;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed
N=10;% number of ensemble members

m=zeros(J,1);v=m;y=m;c=m;U=zeros(J,N);% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=10*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator
U(1,:)=m(1)+sqrt(c(1))*randn(1,N);m(1)=sum(U(1,:))/N;% initial ensemble

%% solution % assimilate!

for j=1:J   
    
    v(j+1)=alpha*sin(v(j)) + sigma*randn;% truth
    y(j)=H*v(j+1)+gamma*randn;% observation

    Uhat=alpha*sin(U(j,:))+sigma*randn(1,N);% ensemble predict
    mhat=sum(Uhat)/N;% estimator predict
    chat=(Uhat-mhat)*(Uhat-mhat)'/(N-1);% covariance predict  

    d=y(j)+gamma*randn(1,N)-H*Uhat;% innovation
    K=(chat*H')/(H*chat*H'+gamma^2);% Kalman gain
    U(j+1,:)=Uhat+K*d;% ensemble update
    m(j+1)=sum(U(j+1,:))/N;% estimator update
    c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))'/(N-1);% covariance update    
    
end

js=21;% plot truth, mean, standard deviation, observations
figure;plot([0:js-1],v(1:js));hold;plot([0:js-1],m(1:js),'m');
plot([0:js-1],m(1:js)+sqrt(c(1:js)),'r--');plot([1:js-1],y(1:js-1),'kx');
plot([0:js-1],m(1:js)-sqrt(c(1:js)),'r--');hold;grid;xlabel('iteration, j');
title('EnKF, Ex. 1.3')

figure;plot([0:J],c);hold
plot([0:J],cumsum(c)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('iteration, j');
title('EnKF Covariance, Ex. 1.3');

figure;plot([0:J],(v-m).^2);hold;
plot([0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('iteration, j');
title('EnKF Error, Ex. 1.3')







