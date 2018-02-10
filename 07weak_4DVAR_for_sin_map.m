function this=p7
clear;set(0,'defaultaxesfontsize',20);format long
%%% p7.m weak 4DVAR for sin map (Ex. 1.3)
%% setup

J=5;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=1e0;% observational noise variance is gamma^2
sigma=1;% dynamics noise variance is sigma^2
C0=1;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed

%% truth

vt(1)=sqrt(C0)*randn;% truth initial condition
for j=1:J
    vt(j+1)=alpha*sin(vt(j))+sigma*randn;% create truth
    y(j)=vt(j+1)+gamma*randn;% create data
end

%% solution

    randn(100);% try uncommenting or changing the argument for different 
               % initial conditions -- if the result is not the same,
               % there may be multimodality (e.g. 1 & 100).                         
    uu=randn(1,J+1);% initial guess       
    %uu=vt;     % truth initial guess option              

% solve with blackbox
% exitflag=1 ==> convergence
[vmap,fval,exitflag]=fminsearch(@(u)I(u,y,sigma,gamma,alpha,m0,C0,J),uu)

figure;plot([0:J],vmap,'Linewidth',2);hold;plot([0:J],vt,'r','Linewidth',2)
plot([1:J],y,'g','Linewidth',2);hold;xlabel('j');legend('MAP','truth','y')

%% auxiliary objective function definition
function out=I(u,y,sigma,gamma,alpha,m0,C0,J)

Phi=0;JJ=1/2/C0*(u(1)-m0)^2;
for j=1:J
    JJ=JJ+1/2/sigma^2*(u(j+1)-alpha*sin(u(j)))^2;
    Phi=Phi+1/2/gamma^2*(y(j)-u(j+1))^2;
end
out=Phi+JJ;




