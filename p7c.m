function this=p7c
clear;set(0,'defaultaxesfontsize',20);format long
%%% p7c.m 4DVAR for inverse Lorenz '63
%% setup

J=0.5e4;% number of steps
a=10;b=8/3;r=28;% define parameters
gamma=1e-1;% observational noise variance is gamma^2
C0=1e-2*eye(3);% prior initial condition covariance
m0=zeros(3,1);% prior initial condition mean
sd=1;rng(sd);% choose random number seed
H=[1,0,0];% observation operator
tau=1e-4;

%% truth

vt(:,1)=m0+sqrtm(C0)*randn(3,1);% truth initial condition
for j=1:J
    vt(:,j+1)=vt(:,j) + tau*f(vt(:,j),a,b,r);% create truth
    dz(:,j)=tau*H*vt(:,j+1)+gamma*sqrt(tau)*randn;% create data
end

%% solution
  
sd=1;rng(sd);% try changing the seed for different 
             % initial conditions -- if the result is not the same,
             % there may be multimodality.                         
    uu=sqrtm(C0)*randn(3,1);% initial guess       
    %uu=vt(:,1);       % truth initial guess option              

% solve with blackbox
% exitflag=1 ==> convergence
[vmap,fval,exitflag]=fminsearch(@(u)I(u,dz,gamma,m0,C0,J,H,a,b,r,tau),uu)

figure;plot(vmap,'ko','Markersize',20, 'Linewidth', 2);%axis([0 4 -1 1.5]);
hold;plot(vt(:,1),'rx','Markersize',20,'Linewidth',2);
hold;xlabel('u');legend('MAP','truth')

%% auxiliary objective function definitions
function out=I(u,dz,gamma,m0,C0,J,H,a,b,r,tau)
Jdet=1/2*(u-m0)'*(C0\(u-m0));Xi3=0;v=zeros(3,J);v(:,1)=u;
for j=1:J
    v(:,j+1)=v(:,j)+tau*f(v(:,j),a,b,r);
    Xi3=Xi3+1/gamma^2*(norm(H*v(:,j+1))^2/2*tau - dz(j)'*H*v(:,j+1));
end
out=Xi3+Jdet;

function rhs=f(y,a,b,r)
rhs(1,1)=a*(y(2)-y(1));
rhs(2,1)=-a*y(1)-y(2)-y(1)*y(3);
rhs(3,1)=y(1)*y(2)-b*y(3)-b*(r+a);



