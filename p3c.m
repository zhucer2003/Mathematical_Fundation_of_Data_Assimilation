clear;set(0,'defaultaxesfontsize',20);format long
%%% p3c.m - smoothing problem for continuous time double-well
%%  setup

C0=5;% variance of the prior
m0=-4;% mean of the prior
sd=1;rng(sd);% choose random number seed
T=10; tau=0.01; t=[0:tau:T]; N=length(t);% time discretization
gamma=1;% observational noise variance is gamma^2

%% truth

vt=zeros(N,1);  z=zeros(N,1);% preallocate space to save time
vt(1)=0.5; z(1)=0;% truth initial condition
dW=sqrt(tau)*randn(N-1,1);% precalculate the Brownian increments used
    for i=1:N-1
        % can be replaced Psi for each problem
        vt(i+1)=vt(i)+tau*(vt(i)-vt(i)^3);  
        z(i+1)=z(i)+tau*vt(i)+gamma*dW(i);% create data
   end

%% solution

v0=[-10:0.01:10];% construct vector of different initial data
Xi3=zeros(length(v0),1); Idet=Xi3; Jdet=Xi3;% preallocate space to save time
for j=1:length(v0) 
    vv=zeros(N,1); vv(1)=v0(j);
   Jdet(j)=1/2/C0*(v0(j)-m0)^2;% background penalization
   for i=1:N-1
        vv(i+1)=vv(i)+tau*(vv(i)-vv(i)^3); 
        Xi3(j)=Xi3(j)+(tau*0.5*vv(i)^2-vv(i)*(z(i+1)-z(i)))/gamma^2;
   end
       Idet(j)=Jdet(j)+Xi3(j);
end


constant=trapz(v0,exp(-Idet));% approximate normalizing constant
P=exp(-Idet)/constant;% normalize posterior distribution
prior=normpdf(v0,m0,C0); % calculate prior distribution    

figure(1),plot(v0,prior,'k','LineWidth',2)
hold on, plot(v0,P,'r--','LineWidth',2), xlabel 'v_0',
legend 'prior' T=10  