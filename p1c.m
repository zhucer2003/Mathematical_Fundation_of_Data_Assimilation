clear; set(0,'defaultaxesfontsize',20); format long
%%% p1c.m behaviour of the double well potential with noise

tau=0.01; T=5e4; t=[0:tau:T];% set up integration constants
x0=3; eps=0.08; Sigma=sqrt(2*eps);% set the SDE coefficients
N=length(t); x=zeros(1,N); x(1)=x0; % set initial conditions
sd=0;rng(sd);% choose random number seed
dW=sqrt(tau)*randn(N-1,1);% precalculate the Brownian increments used

% Euler implementation of the OU
for i=1:N-1
    x(i+1)=x(i)+tau*x(i)*(1-x(i)^2)+Sigma*dW(i);
end

dx=0.01;z=[-5:dx:5]; V=hist(x,z);
p=exp(-0.25*eps^-1*(1-z.^2).^2); p1=p/trapz(z,p);

figure(1), plot(t,x,'k','LineWidth',2)
axis([0 1000 -3 3])
figure(2), plot(z,V./(dx*sum(V)),'r',z,p1,'k','LineWidth',2)
axis([-2 2 0 1.5])
legend 'empirical measure' 'invariant measure'
