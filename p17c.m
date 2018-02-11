function this=p17c
clear;set(0,'defaultaxesfontsize',20);format long
%%% p17c.m  ExKF, Lorenz 96
%% setup

q=2;r=3;n=2;N=r*n;% observe q/r coordinates in N dimensions
J=5e4;F=8;% number of steps and parameter
gamma=1e-1;% observational noise variance is gamma^2
sigma=1e0;% dynamics noise variance is sigma^2
I=eye(N);C0=I;% prior initial condition covariance
m0=zeros(N,1);% prior initial condition mean
sd=1;rng(sd);% choose random number seed

m=zeros(N,J);v=m;z=m(1:q*n,:);c=zeros(N,N,J);% pre-allocate
v(:,1)=m0+sqrtm(C0)*randn(N,1);% initial truth
m(:,1)=10*randn(N,1);% initial mean/estimate
c(:,:,1)=25*C0;% initial covariance operator
H=zeros(q*n,N);for k=1:n;H(q*(k-1)+1:q*k,r*(k-1)+1:r*k)= ...
        [eye(q),zeros(q,r-q)];end% observation operator 
tau=1e-4;% time discretization is tau

%% solution % assimilate!
for j=1:J 
    v(:,j+1)=v(:,j)+tau*f(v(:,j),F) + sigma*sqrt(tau)*randn(N,1);% truth
    z(:,j+1)=z(:,j)+tau*H*v(:,j+1) + gamma*sqrt(tau)*randn(q*n,1);% observation
    mhat=m(:,j)+tau*f(m(:,j),F);% estimator predict
    chat=(I+tau*Df(m(:,j),N))*c(:,:,j)* ...
        (I+tau*Df(m(:,j),N))'+sigma^2*tau*I;% covariance predict      
    d=(z(:,j+1)-z(:,j))/tau-H*mhat;% innovation
    K=(tau*chat*H')/(H*chat*H'*tau+gamma^2*eye(q*n));% Kalman gain
    m(:,j+1)=mhat+K*d;% estimator update
    c(:,:,j+1)=(I-K*H)*chat;% covariance update
end

figure;js=j;
plot(tau*[0:js-1],v(2,1:js));hold;plot(tau*[0:js-1],m(2,1:js),'m');
plot(tau*[0:js-1],m(2,1:js)+reshape(sqrt(c(2,2,1:js)),1,js),'r--');
plot(tau*[0:js-1],m(2,1:js)-reshape(sqrt(c(2,2,1:js)),1,js),'r--');
hold;grid;xlabel('t');legend('u_2','m_2','m_2 \pm c_2^{1/2}')
title('ExKF, L96, 2/3 observed');Jj=min(J,j);
figure;plot(tau*[0:Jj-1],sum((v(:,1:Jj)-m(:,1:Jj)).^2/N));hold;
grid;hold;xlabel('t');title('ExKF, L96, MSE, 2/3 observed')

function rhs=f(y,F)
rhs=[y(end);y(1:end-1)].*([y(2:end);y(1)] - ...
    [y(end-1:end);y(1:end-2)]) - y + F*y.^0;
function A=Df(y,N)
A=-eye(N);
A=A+diag([y(end);y(1:end-2)],1);A(end,1)=y(end-1);
A=A+diag(y(2:end-1),-2);A(1,end-1)=y(end);A(2,end)=y(1);
A=A+diag(([y(3:end);y(1)] - [y(end);y(1:end-2)]),-1);
A(1,end)=y(2)-y(end-1);








