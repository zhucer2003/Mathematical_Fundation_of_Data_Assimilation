function this=p16
clear;set(0,'defaultaxesfontsize',20);format long
%%% p16.m Lorenz '63 (Ex. 2.6)
%% setup

a=10;b=8/3;r=28;% define parameters
sd=1;rng(sd);% choose random number seed

initial=randn(3,1);% choose initial condition
initial1=initial + [0.0001;0;0];% choose perturbed initial condition

%% calculate the trajectories with blackbox  
[t1,y]=ode45(@(t,y) lorenz63(t,y,a,b,r), [0 100], initial);
[t,y1]=ode45(@(t,y) lorenz63(t,y,a,b,r), t1, initial1);

error=sqrt(sum((y-y1).^2,2));% calculate error

%% plot results

figure(1), semilogy(t,error,'k')
axis([0 100 10^-6 10^2])
set(gca,'YTick',[10^-6 10^-4 10^-2 10^0 10^2])

figure(2), plot(t,y(:,1),'k')
axis([0 100 -20 20])


%% auxiliary dynamics function definition
function rhs=lorenz63(t,y,a,b,r)

rhs(1,1)=a*(y(2)-y(1));
rhs(2,1)=-a*y(1)-y(2)-y(1)*y(3);
rhs(3,1)=y(1)*y(2)-b*y(3)-b*(r+a);


