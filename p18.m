clear all
set(0,'defaultaxesfontsize',20)
format long

%setting up the parameters of the problem
a=10;
b=8/3;
r=28;
rng('default'); %always using the same random numbers
initial=randn(3,1);
initial1=initial;
initial1(1)=initial(1)+0.0001;

%calculating the 
[t1,y]=ode45(@(t,y) p16(t,y,a,b,r), [0 100], initial);
[t,y1]=ode45(@(t,y) p16(t,y,a,b,r), t1, initial1);

for i=1:length(t)
error(i)=norm(y(i,:)-y1(i,:),2);
end


figure(1), semilogy(t,error,'b')
axis([0 100 10^-6 10^2])
set(gca,'YTick',[10^-6 10^-4 10^-2 10^0 10^2])
%set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})