clear;set(0,'defaultaxesfontsize',20);format long
%%% p5.m MCMC pCN Dynamics algorithm for
%%% sin map (Ex. 1.3) with noise
%% setup

J=10;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=1;% observational noise variance is gamma^2
sigma=1;% dynamics noise variance is sigma^2
C0=1;% prior initial condition variance
m0=0;% prior initial condition mean
sd=10;rng(sd);% Choose random number seed
 
%% truth

ut=[sqrt(C0)*randn,sigma*randn(1,J)];% truth noise sequence
vt(1)=ut(1);% truth initial condition 
Phi=0;
for j=1:J
    vt(j+1)=alpha*sin(vt(j))+ut(j+1);% create truth
    y(j)=vt(j+1)+gamma*randn;% create data
    % calculate log likelihood phi(u;y) from (1.11)
    Phi=Phi+1/2/gamma^2*(y(j)-vt(j+1))^2;
end

%% solution
% Markov Chain Monte Carlo: N forward steps of the
% Markov Chain on R^{J} with truth initial condition
N=1e5;% number of samples
V=zeros(N,J+1);
beta=0.2;% step-size of pCN walker 
u=ut;v=vt;% truth initial condition (or update Phi)
n=1; bb=0; rat=0;
m=[m0,zeros(1,J)];
while n<=N 
    iota=[sqrt(C0)*randn,sigma*randn(1,J)];% Gaussian prior sample        
    w=m+sqrt(1-beta^2)*(u-m)+beta*iota;% propose sample from the pCN walker 
    vv(1)=w(1);
    Phiprop=0;    
    for j=1:J
        vv(j+1)=alpha*sin(vv(j))+w(j+1);% create path
        Phiprop=Phiprop+1/2/gamma^2*(y(j)-vv(j+1))^2;
    end
    
    if rand<exp(Phi-Phiprop)% accept or reject proposed sample
        u=w;v=vv;Phi=Phiprop;bb=bb+1;% update the Markov chain
    end    
       rat(n)=bb/n;% running rate of acceptance
       V(n,:)=v;% store the chain
       n=n+1;
end
%plot samples, truth, observations
figure;nn=round(rand*N);plot([0:J],V(nn,:));hold;plot([0:J],vt,'r','Linewidth',2);
plot([1:J],y,'g','Linewidth',2);for b=1:1e3; nn=round(rand*N);plot([0:J],V(nn,:));end
plot([0:J],vt,'r','Linewidth',2);plot([1:J],y,'g','Linewidth',2);hold
xlabel('j');legend('Posterior samples','truth','observations')
% trace plot and histogram
jay=J/2;[rho,ex]=hist(V(:,jay),100);figure;plot(V(1:1e4,jay));hold 
plot(rho*5e3/max(rho),ex,'r');title(strcat('v_{',num2str(jay-1),'}|Y_{',num2str(J),'}'))
legend(strcat('v^{(n)}_{',num2str(jay-1),'}|Y_{',num2str(J),'}'),'histogram');xlabel('n')
figure;plot(rat)
figure;plot(cumsum(V(1:N,1))./[1:N]')
xlabel('samples N')
ylabel('(1/N) \Sigma_{n=1}^N v_0^{(n)}')




