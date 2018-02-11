function this=p16c
clear;set(0,'defaultaxesfontsize',20);format long
%%% p16c.m  ExKF, Lorenz 63
%% setup

J=1e5;% number of steps
a=10;b=8/3;r=28;% define parameters
gamma=2e-1;% observational noise variance is gamma^2
sigma=2e0;% dynamics noise variance is sigma^2
I=eye(3);C0=I;% prior initial condition covariance
m0=zeros(3,1);% prior initial condition mean
sd=1;rng(sd);% choose random number seed

m=zeros(3,J);v=m;z=m(1,:);z(1)=0;c=zeros(3,3,J);% pre-allocate
v(:,1)=m0+sqrtm(C0)*randn(3,1);% initial truth
m(:,1)=10*randn(3,1);% initial mean/estimate
c(:,:,1)=10*C0;% initial covariance operator
H=[0,0,1];% observation operator
%H=[1,0,0];% observation operator
tau=1e-4;% time discretization is tau

%% solution % assimilate!
for j=1:J     
    v(:,j+1)=v(:,j)+tau*f(v(:,j),a,b,r) + sigma*sqrt(tau)*randn(3,1);% truth
    z(:,j+1)=z(:,j)+tau*H*v(:,j+1) + gamma*sqrt(tau)*randn;% observation
    mhat=m(:,j)+tau*f(m(:,j),a,b,r);% estimator predict
    chat=(I+tau*Df(m(:,j),a,b))*c(:,:,j)* ...
        (I+tau*Df(m(:,j),a,b))'+sigma^2*tau*I;% covariance predict  
    d=(z(j+1)-z(j))/tau-H*mhat;% innovation
    K=(tau*chat*H')/(H*chat*H'*tau+gamma^2);% Kalman gain     
    m(:,j+1)=mhat+K*d;% estimator update
    c(:,:,j+1)=(I-K*H)*chat;% covariance update         
end

figure;js=j;
plot(tau*[0:js-1],v(2,1:js));hold;plot(tau*[0:js-1],m(2,1:js),'m');
plot(tau*[0:js-1],m(2,1:js)+reshape(sqrt(c(2,2,1:js)),1,js),'r--');
plot(tau*[0:js-1],m(2,1:js)-reshape(sqrt(c(2,2,1:js)),1,js),'r--');
hold;grid;xlabel('t');legend('u_2','m_2','m_2 \pm c_2^{1/2}')
title('ExKF, L63, u_1 observed');Jj=min(J,j);
figure;plot(tau*[0:Jj-1],sum((v(:,1:Jj)-m(:,1:Jj)).^2));hold;
grid;hold;xlabel('t');title('ExKF, L63, MSE, u_1 observed')

function rhs=f(y,a,b,r)
rhs(1,1)=a*(y(2)-y(1));
rhs(2,1)=-a*y(1)-y(2)-y(1)*y(3);
rhs(3,1)=y(1)*y(2)-b*y(3)-b*(r+a);
function A=Df(y,a,b)
A(1,:)=[-a,a,0];
A(2,:)=[-a-y(3),-1,-y(1)];
A(3,:)=[y(2),y(1),-b];








