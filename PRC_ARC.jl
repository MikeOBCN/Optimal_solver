using ForwardDiff, LinearAlgebra, OrdinaryDiffEq, Roots, Zygote, DifferentialEquations

D = 0.3                                                              # parameters
tm = 10.
td = 10.
J= 21.
I = 4.
args = [ D, tm, td, J, I ]

t0 = 0.0 
g0 = [  -1.0023892961642356; 0.0038349864836921266; 0.017262504721870717 ]          #initialization

Z0 = [ 0., 0., 0.]
Y0 = [ g0; Z0 ]
e11 = [ g0; 1.; 0.; 0. ]
e22 = [ g0; 0.; 1.; 0. ]
e33 = [ g0; 0.; 0.; 1. ]

F(x, args) = [ 1/tm*(x[1]^2 - (tm*pi*x[2])^2 - J*tm*x[3] + I);                       #V
               D/(pi*tm^2) + (2/tm)*x[1]*x[2];                                       #Vector Field r
               1/td*(-x[3] + x[2]) ]                                                 #S

t_fire = 18.389055653402046                                                          #mean firing time
t_inhib = 21.343199018077197                                                         #Inhibtion spike
t_inhibmin = 9.54892252628826                                                        #Minimum inhibition

 
function Ad(t, x, y, args)                                                            #Adjoint system
         foo = x -> F(x, args)
         DFt = x -> transpose(ForwardDiff.jacobian(foo, x))
         Ad = -DFt(x)*y
    return Ad
end

function Ad!(dx, x, args, t) 
        dx[1:3] = F(x[1:3], args)
        dx[4:6] = Ad(t, x[1:3], x[4:6], args)
    end 


function Z(t0, Y0, t, args)                                                           #Solving ODE problem and it's adjoint system
        ode1 = ODEProblem(Ad!, convert.(typeof(t),Y0), (t0, t), args)
        z = OrdinaryDiffEq.solve(ode1, Tsit5(), abstol=1e-12, reltol=1e-12)
        return z[end]
end

 ode2 = ODEProblem(Ad!, Y0, (t0, 100), args)
 sol= solve(ode2, Tsit5(), abstol=1e-12, reltol=1e-12);

g(t) = sol(t)[1:3]                                                                    #The period orbit
period(t) = g(t)[1] - g0[1]
T = find_zero(period, (33., 44.))                                                     #Finding the period

om = 1/T
R(t) = [ Z(t0, e11, t, args)[4:6] Z(t0, e22, t, args)[4:6] Z(t0, e33, t, args)[4:6] ] #The resolvant of the adjoint equation
I3 = [ 1 0 0; 0 1 0; 0 0 1 ]
A = R(T)-I3
Z0 = eigvecs(A)[:,1]                                                                  #Periodic intial condition
Y0 = [ g0; Z0 ]
 ode2 = ODEProblem(Ad!, Y0, (t0, 10*T), args)
 sol= solve(ode2, Tsit5(), abstol=1e-12, reltol=1e-12);
PRCi(t)= (1/dot(Z0,F(g0,args)))*sol(t)[4]                                             #In the voltage coordinate 
dPRCi(t) = ForwardDiff.derivative(PRCi, t)
PRC(t) = PRCi(mod(t,T))
#dPRC(t) = dPRCi(mod(t,T))
dPRC(t) = ForwardDiff.derivative(PRC, t)



### ARCs ###

function Lin(t, x, y, args)                                                          #Variational system
         foo = x -> F(x, args)
         DF = x -> ForwardDiff.jacobian(foo, x)
         Lin = DF(x)*y
    return Lin
end
function Lin!(dx, x, args, t) 
        dx[1:3] = F(x[1:3], args)
        dx[4:6] = Lin(t, x[1:3], x[4:6], args)
    end 


ode3 = ODEProblem(Lin!, e11, (t0, 5*T), args)
Mon1 = OrdinaryDiffEq.solve(ode3, Tsit5(), abstol=1e-12, reltol=1e-12)
ode4 = ODEProblem(Lin!, e22, (t0, 5*T), args)
Mon2 = OrdinaryDiffEq.solve(ode4, Tsit5(), abstol=1e-12, reltol=1e-12)
ode5 = ODEProblem(Lin!, e33, (t0, 5*T), args)
Mon3 = OrdinaryDiffEq.solve(ode5, Tsit5(), abstol=1e-12, reltol=1e-12);


MON(t) = [ Mon1(t)[4:6] Mon2(t)[4:6] Mon3(t)[4:6] ]                                 # Monodromy Matrix
mu1 = eigvals(MON(T))[1]
mu2 = eigvals(MON(T))[2] 
lambda1=1/T*log(mu1);                                                               #Floquet Exponants
lambda2=1/T*log(mu2);

v1 = eigvecs(MON(T))[:,1]
v2 = eigvecs(MON(T))[:,2]

K10(t) = MON(t)*exp(-lambda1*t)*v1
K01(t) = MON(t)*exp(-lambda2*t)*v2
dg(t) = ForwardDiff.derivative(sol, t)[1:3]
dg1(t) = dg(t)[1]                                                                    #K0'

ARCl1(t) = cross(K01(t),dg(t))/det([dg(t) K10(t) K01(t)])
ARCl2(t) = cross(dg(t),K10(t))/det([dg(t) K10(t) K01(t)]) 
ARC1(t) = ARCl1(mod(t,T))[1]                                                         #ARC in the Ve direction
ARC2(t) = ARCl2(mod(t,T))[1]                                                         #ARC in the Vi direction
dARCl1(t) = ForwardDiff.derivative(ARCl1, t)
dARCl2(t) = ForwardDiff.derivative(ARCl2, t)
dARC1(t) = dARCl1(mod(t,T))[1]
dARC2(t) = dARCl2(mod(t,T))[1]
