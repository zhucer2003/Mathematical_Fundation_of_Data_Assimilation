clear;set(0,'defaultaxesfontsize',20);format long
%%% p9.m 3DVAR Filter, deterministic logistic map (Ex. 1.4)
%% setup

J=1e3;% number of steps
r=4;% dynamics determined by r
gamma=1e-1;% observational noise variance is gamma^2
sigma=0;% dynamics noise variance is sigma^2
sd=10;rng(sd);% choose random number seed

m=zeros(J,1);v=m;y=m;% pre-allocate
v(1)=rand;% initial truth, in [0,1]
m(1)=rand;% initial mean/estimate, in [0,1]
eta=2e-1;% stabilization coefficient 0 < eta << 1
C=gamma^2/eta;H=1;% covariance and observation operator
K=(C*H')/(H*C*H'+gamma^2);% Kalman gain

%% solution % assimilate!

for j=1:J    
    v(j+1)=r*v(j)*(1-v(j)) + sigma*randn;% truth
    y(j)=H*v(j+1)+gamma*randn;% observation

    mhat=r*m(j)*(1-m(j));% estimator predict
    
    d=y(j)-H*mhat;% innovation
    m(j+1)=mhat+K*d;% estimator update
    
    if norm(mhat)>1e5
        disp('blowup!')
        break
    end
end
js=21;% plot truth, mean, standard deviation, observations
figure;plot([0:js-1],v(1:js));hold;plot([0:js-1],m(1:js),'m');
plot([0:js-1],m(1:js)+sqrt(C),'r--');plot([1:js-1],y(1:js-1),'kx');
plot([0:js-1],m(1:js)-sqrt(C),'r--');hold;grid;xlabel('iteration, j');
title('3DVAR Filter, Ex. 1.4')

figure;plot([0:J],C*[0:J].^0);hold
plot([0:J],C*[0:J].^0,'m','Linewidth',2);grid
hold;xlabel('iteration, j');title('3DVAR Filter Covariance, Ex. 1.4');

figure;plot([0:J],(v-m).^2);hold;
plot([0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);grid
hold;xlabel('iteration, j');
title('3DVAR Filter Error, Ex. 1.4')







