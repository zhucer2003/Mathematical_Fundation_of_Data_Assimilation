clear;set(0,'defaultaxesfontsize',20);format long
% p2c.m - smoothing problem for continuous time OU process
%% setup
C0=5;m0=4;% variance and mean of the prior
sd=1;rng(sd);% choose random number seed
T=10; tau=0.01; t=[0:tau:T]; N=length(t);% time discretization
lambda=-0.5;% dynamics determined by lambda
gamma=1;% observational noise variance is gamma^2

%% truth
vt=zeros(N,1);  z=zeros(N,1);% preallocate space to save time
vt(1)=0.5; z(1)=0;% truth initial condition
dW=sqrt(tau)*randn(N-1,1);% precalculating the Brownian increments used
    for i=1:N-1
        % can be replaced Psi for each problem
        vt(i+1)=exp(lambda*tau)*vt(i);% create truth
        z(i+1)=z(i)+tau*vt(i)+gamma*dW(i);% create data
   end

%% solution
v0=[-10:0.01:10];% construct vector of different initial data
Xi3=zeros(length(v0),1); Idet=Xi3; Jdet=Xi3;% preallocate space to save time
for j=1:length(v0) 
   Jdet(j)=1/2/C0*(v0(j)-m0)^2;% background penalization
   Xi3(j)=1/2*v0(j)^2*gamma^-2*(exp(2*lambda*T)-1)/(2*lambda)- ...
    sum(v0(j)*exp(lambda*t(1:end-1)).*diff(z)')/gamma^2;
   Idet(j)=Xi3(j)+Jdet(j);
end

constant=trapz(v0,exp(-Idet));% approximate normalizing constant
P=exp(-Idet)/constant;% normalize posterior distribution
prior=normpdf(v0,m0,C0); % calculate prior distribution

figure(1),plot(v0,prior,'k','LineWidth',2)
hold on, plot(v0,P,'r--','LineWidth',2), xlabel 'v_0',
legend 'prior' T=10^2  