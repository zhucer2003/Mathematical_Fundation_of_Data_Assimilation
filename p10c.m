clear;set(0,'defaultaxesfontsize',20);format long
%%% p10c.m Continuous 3DVAR Filter, double-well
%% setup

J=1e4;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=1e-1;% observational noise variance is gamma^2
sigma=3e-1;% dynamics noise variance is sigma^2
C0=1;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed

m=zeros(J,1);v=m;z=m;z(1)=0;% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=2*randn;% initial mean/estimate
eta=.1;% stabilization coefficient 0 < eta << 1
c=gamma^2/eta;H=1;% covariance and observation operator
tau=0.01;% time discretization is tau
K=tau*(c*H')/(H*c*H'*tau+gamma^2);% Kalman gain

%% solution % assimilate!

for j=1:J        
    v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) + sigma*sqrt(tau)*randn;% truth
    z(j+1)=z(j)+tau*H*v(j+1) + gamma*sqrt(tau)*randn;% observation
    mhat=m(j)+tau*alpha*(m(j)-m(j)^3);% estimator predict
    d=(z(j+1)-z(j))/tau-H*mhat;% innovation
    m(j+1)=mhat+K*d;% estimator update
end
js=201;% plot truth, mean, standard deviation, observations
figure;plot(tau*[0:js-1],v(1:js));hold;plot(tau*[0:js-1],m(1:js),'m');
plot(tau*[0:js-1],m(1:js)+sqrt(c),'r--',tau*[0:js-1],m(1:js)-sqrt(c),'r--');
hold;grid;xlabel('t');title('3DVAR Filter')

figure;plot(tau*[0:J],c*[0:J].^0);hold
plot(tau*[0:J],c*[0:J].^0,'m','Linewidth',2);grid
hold;xlabel('t');title('3DVAR Filter Covariance');

figure;plot(tau*[0:J],(v-m).^2);hold;
plot(tau*[0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('t');title('3DVAR Filter Error')








