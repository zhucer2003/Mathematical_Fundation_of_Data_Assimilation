function this=p2_5
clear all;set(0,'defaultaxesfontsize',20);format long
%%% p2.5.m - behaviour of Lorenz '63 (Ex. 1.6) 

%set up the parameters of the problem
a=10;
b=8/3;
r=28;
sd=1;rng(sd); %always using the same random numbers
initial=randn(3,1);
initial1=initial;
initial1(1)=initial(1)+0.0001;

%calculate the trajectory
[t1,y]=ode45(@(t,y) lorenz63(t,y,a,b,r), [0 100], initial);
[t,y1]=ode45(@(t,y) lorenz63(t,y,a,b,r), t1, initial1);

for i=1:length(t)
error(i)=norm(y(i,:)-y1(i,:),2);
end


figure(1), semilogy(t,error,'b')
axis([0 100 10^-6 10^2])
set(gca,'YTick',[10^-6 10^-4 10^-2 10^0 10^2])
%set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})

function dy=lorenz63(t,y,a,b,r)

dy(1)=a*(y(2)-y(1));
dy(2)=-a*y(1)-y(2)-y(1)*y(3);
dy(3)=y(1)*y(2)-b*y(3)-b*(r+a);

dy=dy';