function this=p17
clear;set(0,'defaultaxesfontsize',20);format long
%%% p17.m Lorenz '96 (Ex. 2.7)
%% setup

J=40;F=8;% define parameters
sd=1;rng(sd);% choose random number seed

initial=randn(J,1);% choose initial condition
initial1=initial;
initial1(1)=initial(1)+0.0001;% choose perturbed initial condition

%% calculate the trajectories with blackbox  
[t1,y]=ode45(@(t,y) lorenz96(t,y,F), [0 100], initial);
[t,y1]=ode45(@(t,y) lorenz96(t,y,F), t1, initial1);

error=sqrt(sum((y-y1).^2,2));% calculate error

%% plot results

figure(1), plot(t,y(:,1),'k')
figure(2), plot(y(:,1),y(:,J),'k')
figure(3), plot(y(:,1),y(:,J-1),'k')

figure(4), semilogy(t,error,'k')
axis([0 100 10^-6 10^2])
set(gca,'YTick',[10^-6 10^-4 10^-2 10^0 10^2])

%% auxiliary dynamics function definition
function rhs=lorenz96(t,y,F)

rhs=[y(end);y(1:end-1)].*([y(2:end);y(1)] - ...
    [y(end-1:end);y(1:end-2)]) - y + F*y.^0;



