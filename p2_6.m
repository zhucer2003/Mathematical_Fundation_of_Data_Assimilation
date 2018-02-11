function this=p2_6
clear all;set(0,'defaultaxesfontsize',20);format long
%%% p2.6.m - behaviour of Lorenz '96 (Ex. 1.7) 

%set up the parameters of the problem
J=40;
F=8;
rng('default'); %always using the same random numbers
initial=randn(J,1);
initial1=initial;
initial1(1)=initial(1)+0.0001;

%calculate the trajectory
[t1,y]=ode45(@(t,y) lorenz96(t,y,F,J), [0 100], initial);
[t,y1]=ode45(@(t,y) lorenz96(t,y,F,J), t1, initial1);

figure(1), plot(t,y(:,1),'b')
figure(2), plot(y(:,1),y(:,J),'b')
figure(3), plot(y(:,1),y(:,J-1),'b')


for i=1:length(t)
error(i)=norm(y(i,:)-y1(i,:),2);
end


figure(4), semilogy(t,error,'b')
axis([0 100 10^-6 10^2])
set(gca,'YTick',[10^-6 10^-4 10^-2 10^0 10^2])
%set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})

function dy=lorenz96(t,y,F,J)

%setting the first two lines 
dy(1)=y(J)*(y(2)-y(J-1))-y(1)+F;
dy(2)=y(1)*(y(3)-y(J))-y(2)+F;
 
%recursion is fine here
for j=3:J-1
    dy(j)=y(j-1)*(y(j+1)-y(j-2))-y(j)+F;
end

%setting the last line
dy(J)=y(J-1)*(y(1)-y(J-2))-y(J)+F;


dy=dy';