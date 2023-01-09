
using Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, NLsolve, Plots

################################################################
#### Defining parameters and other structures for the model ####
################################################################
#Define parameter and grid structure, use quarterly calibration
@with_kw struct Pars
    ################################################################
    ######## Preferences and technology ############################
    ################################################################
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter
    β::Float64 = 0.954         #Discount factor
    r_star::Float64 = 0.01    #Risk-fr  ee interest rate
    #Debt parameters
    γ::Float64 = 0.05         #Reciprocal of average maturity
    κ::Float64 = 0.03         #Coupon payments
    #Default cost and debt parameters
    θ::Float64 = 0.0625 #0.0385       #Probability of re-admission (2 years)
    d0::Float64 = -0.18819#-0.2914       #income default cost
    d1::Float64 = 0.24558#0.4162        #income default cost
    #Capital accumulation
    δ::Float64 = 0.07       #Depreciation rate
    φ::Float64 = 2.5        #Capital adjustment cost
    #Production functions
    #Final consumption good
    η::Float64 = 0.83
    ω::Float64 = 0.60
    A::Float64 = 1.0            #Scaling factor for final good
    E_out::Float64 = η/(η-1.0)
    E_in::Float64 = (η-1.0)/η
    aN::Float64 = ω^(1.0/η)
    aT::Float64 = (1.0-ω)^(1.0/η)
    #Value added of intermediates
    αN::Float64 = 0.33          #Capital share in non-traded sector
    αT::Float64 = 0.33          #Capital share in manufacturing sector
    #Stochastic process
    #Parameters to pin down steady state capital and scaling parameters
    Target_yTss::Float64 = 1.0        #Target value for steady state output
    Target_iss_gdp::Float64 = 0.20 #Target for investment / gdp in steady state
    #parameters for productivity shock
    μ_ϵz::Float64 = 0.0
    σ_ϵz::Float64 = 0.017#0.027
    dist_ϵz::UnivariateDistribution = truncated(Normal(μ_ϵz,σ_ϵz),-2.0*σ_ϵz,2.0*σ_ϵz)
    ρ_z::Float64 = 0.95#0.948
    μ_z::Float64 = 1.0
    zlow::Float64 = exp(log(μ_z)-2.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+2.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #Quadrature parameters
    N_GL::Int64 = 21
    #Grids
    Nz::Int64 = 5
    NkN::Int64 = 11
    NkT::Int64 = 11
    Nb::Int64 = 31
    NbOpt::Int64 = 1000
    NT::Int64 = 7
    kNlow::Float64 = 0.25
    kNhigh::Float64 = 2.0
    kTlow::Float64 = 0.10
    kThigh::Float64 = 0.55
    blow::Float64 = 0.0
    bhigh::Float64 = 6.5
    #Parameters for solution algorithm
    cmin::Float64 = 1e-2
    relTol::Float64 = 0.9       #Tolerance for relative error in VFI (0.1%)
    Tol_V::Float64 = 1e-6       #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-3       #Tolerance for absolute distance for q
    Tolpct_q::Float64 = 1.0        #Tolerance for % of states for which q has not converged
    cnt_max::Int64 = 200           #Maximum number of iterations on VFI
    MaxIter_Opt::Int64 = 1000
    g_tol_Opt::Float64 = 1e-4
    blowOpt::Float64 = blow-1e-2             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.1            #Maximum level of debt for optimization
    kNlowOpt::Float64 = 0.9*kNlow             #Minimum level of capital for optimization
    kNhighOpt::Float64 = 1.1*kNhigh           #Maximum level of capital for optimization
    kTlowOpt::Float64 = 0.9*kTlow            #Minimum level of oil capital for optimization
    kThighOpt::Float64 = 1.1*kThigh           #Maximum level of oil capital for optimization
    #Simulation parameters
    Tsim::Int64 = 10000
    drp::Int64 = 1000
    Tmom::Int64 = 50
    TsinceDefault::Int64 = 25
    NSamplesMoments::Int64 = 300
    HPFilter_Par::Float64 = 100.0
    t_before_crisis::Int64 = 10
    t_after_crisis::Int64 = 10
    N_crises::Int64 = 300
    #Parameters for graphs
    size_width::Int64 = 650
    size_height::Int64 = 500
end

@with_kw struct Grids
    #Grids of states
    GR_z::Array{Float64,1}
    GR_kN::Array{Float64,1}
    GR_kT::Array{Float64,1}
    GR_b::Array{Float64,1}
    GR_bopt::Array{Float64,1}
    #Quadrature vectors for integrals
    GL_nodes::Vector{Float64}
    GL_weights::Vector{Float64}
    ϵz_weights::Vector{Float64}
    ϵz_nodes::Vector{Float64}
    #Factor to correct quadrature underestimation
    FacQ::Float64
    #Bounds for divide and conquer for b
    ind_order_b::Array{Int64,1}
    lower::Array{Int64,1}
    upper::Array{Int64,1}
    #Matrices for integrals
    ZPRIME::Array{Float64,2}
    PDFz::Array{Float64,2}
end

function BoundsDivideAndConquer(N::Int64)
    ind_order=Array{Int64,1}(undef,N)
    lower=Array{Int64,1}(undef,N)
    upper=Array{Int64,1}(undef,N)
    i=1
    #Do for last first
    ind_order[i]=N
    lower[i]=1
    upper[i]=N
    i=i+1
    #Do for first
    ind_order[i]=1
    lower[i]=1
    upper[i]=N
    i=i+1
    #Auxiliary vectors
    d=zeros(Int64,floor(Int64,N/2)+1)
    up=zeros(Int64,floor(Int64,N/2)+1)
    k=1
    d[1]=1
    up[1]=N
    k=1
    while true

        while true

            if up[k]==(d[k]+1)
                break
            end
            k=k+1
            d[k]=d[k-1]
            up[k]=floor(Int64,(d[k-1]+up[k-1])/2)
            # Compute for g of u(k) searching from g(l(k-1)) to g(u(k-1))b
            ind_order[i]=up[k]
            lower[i]=d[k-1]
            upper[i]=up[k-1]
            i=i+1
        end

        while true
            if k==1
                break
            end
            if up[k]!=up[k-1]
                break
            end
            k=k-1
        end
        if k==1
            break
        end
        d[k]=up[k]
        up[k]=up[k-1]
    end
    return ind_order, lower, upper
end

function CreateGrids(par::Pars)
    #Grid for z
    @unpack Nz, zlow, zhigh = par
    GR_z=collect(range(zlow,stop=zhigh,length=Nz))
    #Gauss-Legendre vectors for z
    @unpack N_GL, σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    GL_nodes, GL_weights = gausslegendre(N_GL)
    ϵzlow=-3.0*σ_ϵz
    ϵzhigh=3.0*σ_ϵz
    ϵz_nodes=0.5*(ϵzhigh-ϵzlow).*GL_nodes .+ 0.5*(ϵzhigh+ϵzlow)
    ϵz_weights=GL_weights .* 0.5*(ϵzhigh-ϵzlow)
    #Matrices for integration over z
    ZPRIME=Array{Float64,2}(undef,Nz,N_GL)
    PDFz=Array{Float64,2}(undef,Nz,N_GL)
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        ZPRIME[z_ind,:]=exp.((1.0-ρ_z)*log(μ_z)+ρ_z*log(z) .+ ϵz_nodes)
        PDFz[z_ind,:]=pdf.(dist_ϵz,ϵz_nodes)
    end
    FacQ=dot(ϵz_weights,pdf.(dist_ϵz,ϵz_nodes))
    #Grid of non-traded capital
    @unpack NkN, kNlow, kNhigh = par
    GR_kN=collect(range(kNlow,stop=kNhigh,length=NkN))
    #Grid of traded capital
    @unpack NkT, kTlow, kThigh = par
    GR_kT=collect(range(kTlow,stop=kThigh,length=NkT))
    #Grid of debt
    @unpack Nb, NbOpt, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))
    GR_bopt=collect(range(blow,stop=bhigh,length=NbOpt))
    #Bounds for divide and conquer
    ind_order, lower, upper=BoundsDivideAndConquer(Nb)
    return Grids(GR_z,GR_kN,GR_kT,GR_b,GR_bopt,GL_nodes,GL_weights,ϵz_weights,ϵz_nodes,FacQ,ind_order,lower,upper,ZPRIME,PDFz)
end

@with_kw mutable struct Solution{T1,T2,T3,T4,T5,T6,T7,T8,T9}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    #Policy functions
    kNprime_D::T1
    kTprime_D::T1
    kNprime::T2
    kTprime::T2
    bprime::T2
    Tr::T2
    yD::T1
    yNP::T8
    ccNP::T8
    yTP::T9
    cD::T1
    cP::T2
    ### Interpolation objects
    #Value Functions
    itp_VD::T3
    itp_VP::T4
    itp_V::T4
    #Expectations and price
    itp_EVD::T3
    itp_EV::T4
    itp_q1::T5
    #Policy functions
    itp_kNprime_D::T6
    itp_kTprime_D::T6
    itp_kNprime::T7
    itp_kTprime::T7
    itp_bprime::T7
    itp_Tr::T7
end

function MyBisection(foo,a::Float64,b::Float64;xatol::Float64=1e-8)
    s=sign(foo(a))
    x=(a+b)/2.0
    d=(b-a)/2.0
    while d>xatol
        d=d/2.0
        if s==sign(foo(x))
            x=x+d
        else
            x=x-d
        end
    end
    return x
end

#Interpolate equilibrium objects
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_kN, GR_kT = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()#Cubic(Line(OnGrid()))
    ORDER_B_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KTs,KNs,Zs),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs),Interpolations.Line())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_kN, GR_kT, GR_b = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS))
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs),Interpolations.Flat())
end

function CreateInterpolation_Policies(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_kN, GR_kT = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KTs,KNs,Zs),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs),Interpolations.Flat())
    end
end

function CreateInterpolation_cPol(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_kN, GR_kT = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    KNs=range(GR_kN[1],stop=GR_kN[end],length=length(GR_kN))
    KTs=range(GR_kT[1],stop=GR_kT[end],length=length(GR_kT))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KTs,KNs,Zs),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        ORDER_B_STATES=Linear()
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_K_STATES),BSpline(ORDER_SHOCKS))
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KTs,KNs,Zs),Interpolations.Flat())
    end
end

################################################################
################ Preference functions ##########################
################################################################
function Utility(c::Float64,par::Pars)
    @unpack σ = par
    return (c^(1.0-σ))/(1.0-σ)
end

function zDefault(z::Float64,par::Pars)
    @unpack d0, d1 = par
    return z-max(0.0,d0*z+d1*z*z)
end

function CapitalAdjustment(kprime::Float64,k::Float64,par::Pars)
    @unpack φ, δ = par
    i=kprime-(1.0-δ)k
    0.5*φ*(i^2.0)/k
end

function dΨ_dkprime(kprime::Float64,k::Float64,par::Pars)
    @unpack φ, δ = par
    i=kprime-(1.0-δ)k
    return φ*i/k
end

function dΨ_dk(kprime::Float64,k::Float64,par::Pars)
    @unpack φ, δ = par
    i=kprime-(1.0-δ)k
    return -(φ/2.0)*((i/k)^2.0)
end

function SDF_Lenders(z::Float64,ϵz::Float64,par::Pars)
    @unpack r_star = par
    return exp(-r_star)
end

################################################################
################ Production functions ##########################
################################################################
function NonTradedProduction(z::Float64,kN::Float64,par::Pars)
    @unpack αN, A = par
    return z*A*(kN^αN)
end

function TradedProduction(z::Float64,kT::Float64,par::Pars)
    @unpack αT, A = par
    return z*A*(kT^αT)
end

function Final_CES(cN::Float64,cT::Float64,par::Pars)
    @unpack E_out, E_in, aN, aT = par
    # return ((ω^(1.0/η))*(cN^((η-1.0)/η))+((1.0-ω)^(1.0/η))*(cT^((η-1.0)/η)))^(η/(η-1.0))
    return (aN*(cN^E_in)+aT*(cT^E_in))^E_out
end

function Final_CES_Fast(z_ind::Int64,kN_ind::Int64,cT::Float64,SOLUTION::Solution,par::Pars)
    @unpack E_out, E_in, aT = par
    return (SOLUTION.ccNP[kN_ind,z_ind]+aT*(cT^E_in))^E_out
end

function FinalOutput(z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    #Compute consumption of intermediate goods
    cT=TradedProduction(z,kT,par)+T
    cN=NonTradedProduction(z,kN,par)
    if cT>0.0
        return Final_CES(cN,cT,par)
    else
        return cT
    end
end

function FinalOutputFast(I::CartesianIndex,T::Float64,SOLUTION::Solution,par::Pars)
    #Compute consumption of intermediate goods
    cT=SOLUTION.yTP[I[2],I[4]]+T
    if cT>0.0
        return Final_CES_Fast(I[4],I[3],cT,SOLUTION,par)
    else
        return cT
    end
end

################################################################
################### Setup functions ############################
################################################################
function MPK_N(z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    hh=1e-7
    return (FinalOutput(z,kN+hh,kT,T,par)-FinalOutput(z,kN-hh,kT,T,par))/(2.0*hh)
end

function MPK_T(z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    hh=1e-7
    return (FinalOutput(z,kN,kT+hh,T,par)-FinalOutput(z,kN,kT-hh,T,par))/(2.0*hh)
end

function kTss_Given_kNss(zss::Float64,kNss::Float64,par::Pars)
    @unpack β, δ, μ_z, A = par
    foo(kTss::Float64)=β*(MPK_T(μ_z,kNss,kTss,0.0,par)+1.0-δ)-1.0
    #Find a bracketing interval
    kTlow=0.01
    while foo(kTlow)<=0.0
        kTlow=0.95*kTlow
        if kTlow<1e-6
            break
        end
    end
    kThigh=kNss
    while foo(kThigh)>=0.0
        kThigh=2.0*kThigh
    end
    return MyBisection(foo,kTlow,kThigh;xatol=1e-2)
end

function SteadyStateCapital(par::Pars)
    @unpack β, δ, μ_z = par
    foo(kNss::Float64)=β*(MPK_N(μ_z,kNss,kTss_Given_kNss(μ_z,kNss,par),0.0,par)+1.0-δ)-1.0
    #Find a bracketing interval
    kNlow=1.0
    while foo(kNlow)<=0.0
        kNlow=0.95*kNlow
    end
    kNhigh=3.0
    while foo(kNhigh)>=0.0
        kNhigh=2.0*kNhigh
    end
    kNss=MyBisection(foo,kNlow,kNhigh;xatol=1e-3)
    kTss=kTss_Given_kNss(μ_z,kNss,par)
    return kNss, kTss
end

function PriceNonTraded(z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack αN, αT, ω, η = par
    cT=TradedProduction(z,kT,par)+T
    cN=NonTradedProduction(z,kN,par)
    if cT>0.0
        return ((ω/(1.0-ω))*(cT/cN))^(1.0/η)
    else
        return 1e-2
    end
end

function PriceFinalGood(z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack ω, η, A = par
    pN=PriceNonTraded(z,kN,kT,T,par)
    return (ω*(pN^(1.0-η))+(1.0-ω)*(1.0^(1.0-η)))^(1.0/(1.0-η))
end

function Calibrate_A(par::Pars)
    @unpack Target_yTss, μ_z = par
    function foo(Ass::Float64)
        parA=Pars(par,A=Ass)
        kNss, kTss=SteadyStateCapital(parA)
        yT=TradedProduction(μ_z,kTss,parA)
        return yT-Target_yTss
    end
    #Get bracketing interval
    Alow=0.95
    while foo(Alow)>=0.0
        Alow=0.95*Alow
    end
    Ahigh=1.05
    while foo(Ahigh)<=0.0
        Ahigh=2.0*Ahigh
    end
    A_ss=MyBisection(foo,Alow,Ahigh;xatol=1e-3)
    parA=Pars(par,A=A_ss)
    return A_ss
end

function Calibrate_delta_A(par::Pars)
    @unpack Target_iss_gdp = par
    function foo(δ::Float64)
        par=Pars(par,δ=δ)
        A=Calibrate_A(par)
        par=Pars(par,A=A)
        kNss, kTss=SteadyStateCapital(par)
        yss=FinalOutput(par.μ_z,kNss,kTss,0.0,par)
        return δ*(kNss+kTss)/yss-Target_iss_gdp
    end
    #Get bracketing interval
    δlow=0.01
    while foo(δlow)>=0.0
        δlow=0.95*δlow
    end
    δhigh=0.15
    while foo(δhigh)<=0.0 && δhigh<0.9
        δhigh=δhigh+0.01
    end
    δ=MyBisection(foo,δlow,δhigh;xatol=1e-3)
    par=Pars(par,δ=δ)
    A=Calibrate_A(par)
    return δ, A
end

function Setup(β::Float64,φ::Float64,d0::Float64,d1::Float64)
    par=Pars(β=β,φ=φ,d0=d0,d1=d1)
    #Calibrate A and δ
    δ, A=Calibrate_delta_A(par)
    par=Pars(par,δ=δ,A=A)
    #Setup Grids range
    kNss, kTss=SteadyStateCapital(par)
    par=Pars(par,kNlow=0.25*kNss,kNhigh=2.0*kNss)
    par=Pars(par,kTlow=0.25*kTss,kThigh=2.0*kTss)
    GRIDS=CreateGrids(par)
    #Set bounds for optimization algorithm
    par=Pars(par,kNlowOpt=0.5*par.kNlow,kNhighOpt=1.1*par.kNhigh)
    par=Pars(par,kTlowOpt=0.5*par.kTlow,kThighOpt=1.1*par.kThigh)
    return par, GRIDS
end

###############################################################################
#Function to compute consumption net of investment and adjustment cost
###############################################################################
function ConsNet(y::Float64,kN::Float64,kT::Float64,kNprime::Float64,kTprime::Float64,par::Pars)
    @unpack δ = par
    iN=kNprime-(1.0-δ)kN
    iT=kTprime-(1.0-δ)kT
    y-iN-iT-CapitalAdjustment(kNprime,kN,par)-CapitalAdjustment(kTprime,kT,par)
end

###############################################################################
#Functions to compute value given state, policies, and guesses
###############################################################################
# transform function
function TransformIntoBounds(x::Float64,min::Float64,max::Float64)
    (max - min) * (1.0/(1.0 + exp(-x))) + min
end

function TransformIntoReals(x::Float64,min::Float64,max::Float64)
    log((x - min)/(max - x))
end

function ValueInDefault(z::Float64,kN::Float64,kT::Float64,y::Float64,
                        kNprime_REAL::Float64,kTprime_REAL::Float64,
                        VDmin::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt, cmin = par
    @unpack GR_kN, GR_kT = GRIDS
    @unpack itp_EV, itp_EVD = SOLUTION
    #transform policy tries into interval
    kNprime=TransformIntoBounds(kNprime_REAL,kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(kTprime_REAL,kTlowOpt,kThighOpt)
    #Compute consumption and value
    cons=ConsNet(y,kN,kT,kNprime,kTprime,par)
    if cons>0.0
        return Utility(cons,par)+β*θ*min(0.0,itp_EV(0.0,kTprime,kNprime,z))+β*(1.0-θ)*min(0.0,itp_EVD(kTprime,kNprime,z))
    else
        return Utility(cmin,par)-kNprime-kTprime
    end
end

function ValueInRepayment(z::Float64,kN::Float64,kT::Float64,b::Float64,I::CartesianIndex,
                          kNprime_REAL::Float64,kTprime_REAL::Float64,bprime_REAL::Float64,
                          SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, κ, kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt, blowOpt ,bhighOpt, cmin = par
    @unpack itp_EV, itp_q1 = SOLUTION
    @unpack GR_kN, GR_kT = GRIDS
    #transform policy tries into interval
    kNprime=TransformIntoBounds(kNprime_REAL,kNlowOpt,kNhighOpt)
    kTprime=TransformIntoBounds(kTprime_REAL,kTlowOpt,kThighOpt)
    bprime=TransformIntoBounds(bprime_REAL,blowOpt,bhighOpt)
    #Compute output
    qq=itp_q1(bprime,kTprime,kNprime,z)
    T=qq*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
    aa=0.0
    if qq==0.0
        aa=-bprime
    end
    # y=FinalOutput(z,kN,kT,T,par)
    y=FinalOutputFast(I,T,SOLUTION,par)
    #Compute consumption
    cons=ConsNet(y,kN,kT,kNprime,kTprime,par)
    if y>0.0 && cons>cmin
        return Utility(cons,par)+β*min(0.0,itp_EV(bprime,kTprime,kNprime,z))+aa
    else
        return Utility(cmin,par)-kNprime-kTprime+aa
    end
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################

function OptimInDefault!(I::CartesianIndex,VDmin::Float64,
                        SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, δ, MaxIter_Opt, g_tol_Opt, kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt = par
    @unpack GR_z, GR_kN, GR_kT = GRIDS
    (kT_ind,kN_ind,z_ind)=Tuple(I)
    #Use previous solution as initial guess
    X0_BOUNDS=[GR_kN[kN_ind]; GR_kT[kT_ind]]
    #transform policy guess into reals
    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],kNlowOpt,kNhighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kTlowOpt,kThighOpt)
    #Setup function handle for optimization
    z=GR_z[z_ind]
    zD=zDefault(z,par)
    kN=GR_kN[kN_ind]
    kT=GR_kT[kT_ind]
    y=SOLUTION.yD[I]#FinalOutput(zD,kN,kT,0.0,par)
    f(X::Array{Float64,1})=-ValueInDefault(z,kN,kT,y,X[1],X[2],VDmin,SOLUTION,GRIDS,par)
    #Perform optimization
    inner_optimizer = NelderMead()
    res=optimize(f,X0,inner_optimizer)
    #Transform optimizer into bounds
    SOLUTION.kNprime_D[I]=TransformIntoBounds(Optim.minimizer(res)[1],kNlowOpt,kNhighOpt)
    SOLUTION.kTprime_D[I]=TransformIntoBounds(Optim.minimizer(res)[2],kTlowOpt,kThighOpt)
    SOLUTION.VD[I]=-Optim.minimum(res)
    return nothing
end

function OptimInRepayment!(I::CartesianIndex,VPmin::Float64,
                           SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, δ, MaxIter_Opt, g_tol_Opt, Nb, kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt, blowOpt, bhighOpt = par
    @unpack GR_kN, GR_kT, GR_b, GR_z = GRIDS
    z=GR_z[I[4]]
    kN=GR_kN[I[3]]
    kT=GR_kT[I[2]]
    b=GR_b[I[1]]
    #transform policy guess into reals
    X0=Array{Float64,1}(undef,3)
    X0[1]=TransformIntoReals(kN,kNlowOpt,kNhighOpt)
    X0[2]=TransformIntoReals(kT,kTlowOpt,kThighOpt)
    X0[3]=TransformIntoReals(b,blowOpt,bhighOpt)
    #Setup function handle for optimization
    f(X::Array{Float64,1})=-ValueInRepayment(z,kN,kT,b,I,X[1],X[2],X[3],SOLUTION,GRIDS,par)
    #Perform optimization with MatLab simplex
    inner_optimizer = NelderMead()
    res=optimize(f,X0,inner_optimizer)
    #Transform optimizer into bounds
    SOLUTION.kNprime[I]=TransformIntoBounds(Optim.minimizer(res)[1],kNlowOpt,kNhighOpt)
    SOLUTION.kTprime[I]=TransformIntoBounds(Optim.minimizer(res)[2],kTlowOpt,kThighOpt)
    SOLUTION.bprime[I]=TransformIntoBounds(Optim.minimizer(res)[3],blowOpt,bhighOpt)
    SOLUTION.VP[I]=-Optim.minimum(res)
    return nothing
end

###############################################################################
#Update default
###############################################################################

function Expectation_Default!(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, dist_ϵz = par
    @unpack ϵz_weights, ZPRIME, PDFz, FacQ = GRIDS
    @unpack itp_VD = SOLUTION
    (kT_ind,kN_ind,z_ind)=Tuple(I)
    kNprime=GRIDS.GR_kN[kN_ind]
    kTprime=GRIDS.GR_kT[kT_ind]
    funVD(zprime::Float64)=min(0.0,itp_VD(kTprime,kNprime,zprime))
    pdf_VD=PDFz[z_ind,:] .* funVD.(ZPRIME[z_ind,:])
    SOLUTION.EVD[I]=dot(ϵz_weights,pdf_VD)/FacQ
    return nothing
end

function UpdateDefault!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT = par
    @unpack GR_kN, GR_kT, GR_z = GRIDS
    #Loop over all states to fill array of VD
    VDmin=minimum(SOLUTION.VD)
    for I in CartesianIndices(SOLUTION.VD)
        OptimInDefault!(I,VDmin,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_kNprime_D=CreateInterpolation_Policies(SOLUTION.kNprime_D,true,GRIDS)
    SOLUTION.itp_kTprime_D=CreateInterpolation_Policies(SOLUTION.kTprime_D,true,GRIDS)
    #Loop over all states to compute expectations over p and z
    for I in CartesianIndices(SOLUTION.EVD)
        Expectation_Default!(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

###############################################################################
#Update repayment
###############################################################################

function Expectation_Repayment!(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, dist_ϵz = par
    @unpack ϵz_weights, ZPRIME, PDFz, FacQ = GRIDS
    @unpack itp_V = SOLUTION
    (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
    kNprime=GRIDS.GR_kN[kN_ind]
    kTprime=GRIDS.GR_kT[kT_ind]
    bprime=GRIDS.GR_b[b_ind]
    funV(zprime::Float64)=min(0.0,itp_V(bprime,kTprime,kNprime,zprime))
    pdf_VD=PDFz[z_ind,:] .* funV.(ZPRIME[z_ind,:])
    SOLUTION.EV[I]=dot(ϵz_weights,pdf_VD)/FacQ
    return nothing
end

function UpdateRepayment!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb, γ, κ = par
    @unpack GR_kN, GR_kT, GR_b, GR_z = GRIDS
    @unpack kNprime_D, kTprime_D, itp_q1 = SOLUTION
    VPmin=minimum(SOLUTION.VP)
    #Allocate shared arrays
    #Loop over all states to fill value of repayment
    for I in CartesianIndices(SOLUTION.VP)
        (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
        OptimInRepayment!(I,VPmin,SOLUTION,GRIDS,par)
        if SOLUTION.VP[I]<SOLUTION.VD[kT_ind,kN_ind,z_ind]
            SOLUTION.V[I]=SOLUTION.VD[kT_ind,kN_ind,z_ind]
            SOLUTION.Tr[I]=0.0
        else
            SOLUTION.V[I]=SOLUTION.VP[I]
            qq=itp_q1(SOLUTION.bprime[I],SOLUTION.kTprime[I],SOLUTION.kNprime[I],GR_z[z_ind])
            SOLUTION.Tr[I]=qq*(SOLUTION.bprime[I]-(1.0-γ)*GR_b[b_ind])-(γ+κ*(1.0-γ))*GR_b[b_ind]
        end
    end
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kNprime=CreateInterpolation_Policies(SOLUTION.kNprime,false,GRIDS)
    SOLUTION.itp_kTprime=CreateInterpolation_Policies(SOLUTION.kTprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,false,GRIDS)
    SOLUTION.itp_Tr=CreateInterpolation_Policies(SOLUTION.Tr,false,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    for I in CartesianIndices(SOLUTION.EV)
        Expectation_Repayment!(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

###############################################################################
#Update price
###############################################################################
function BondsPayoff(ϵz::Float64,z::Float64,kNprime::Float64,kTprime::Float64,
                     bprime::Float64,SOLUTION::Solution,par::Pars)
    @unpack ρ_z, μ_z, γ, κ = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime, itp_kNprime, itp_kTprime = SOLUTION
    zprime=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z)+ϵz)
    if min(0.0,itp_VD(kTprime,kNprime,zprime))>min(0.0,itp_VP(bprime,kTprime,kNprime,zprime))
        return 0.0
    else
        SDF=SDF_Lenders(z,ϵz,par)
        kNkN=itp_kNprime(bprime,kTprime,kNprime,zprime)
        kTkT=itp_kTprime(bprime,kTprime,kNprime,zprime)
        bb=itp_bprime(bprime,kTprime,kNprime,zprime)
        return SDF*(γ+(1.0-γ)*(κ+itp_q1(bb,kTkT,kNkN,zprime)))
    end
end

function Integrate_BondsPrice!(I::CartesianIndex,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵz = par
    @unpack ϵz_nodes, ϵz_weights, FacQ = GRIDS
    (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
    z=GRIDS.GR_z[z_ind]
    kNprime=GRIDS.GR_kN[kN_ind]
    kTprime=GRIDS.GR_kT[kT_ind]
    bprime=GRIDS.GR_b[b_ind]
    fun(ϵz::Float64)=pdf(dist_ϵz,ϵz)*BondsPayoff(ϵz,z,kNprime,kTprime,bprime,SOLUTION,par)
    SOLUTION.q1[I]=dot(ϵz_weights,fun.(ϵz_nodes))/FacQ
    return nothing
end

function UpdateBondsPrice!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb = par
    @unpack GR_z, GR_kN, GR_kT, GR_b = GRIDS
    #Loop over all states to compute expectation over z' and p'
    for I in CartesianIndices(SOLUTION.q1)
        Integrate_BondsPrice!(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    return nothing
end

###############################################################################
#Update and iniciate VFI algorithm
###############################################################################
function UpdateSolution!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault!(SOLUTION,GRIDS,par)
    UpdateRepayment!(SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    return nothing
end

function ComputeDistance_q(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    @unpack Tol_q = par
    dst_q=maximum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1))
    NotConv=sum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.q1)
    return round(dst_q,digits=5), round(NotConvPct,digits=2)
end

function ComputeDistanceV(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution)
    dst_D=maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD))
    dst_V=maximum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V))
    return round(abs(dst_D),digits=5), round(abs(dst_V),digits=5)
end

function ComputeRelativeDistance(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution)
    dst_D=100.0*maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD) ./ abs.(SOLUTION_CURRENT.VD))
    dst_V=100.0*maximum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V) ./ abs.(SOLUTION_CURRENT.V))
    return round(abs(dst_D),digits=4), round(abs(dst_V),digits=4)
end

function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb, γ, κ = par
    ### Allocate all values to object
    VD=zeros(Float64,NkT,NkN,Nz)
    VP=zeros(Float64,Nb,NkT,NkN,Nz)
    V=zeros(Float64,Nb,NkT,NkN,Nz)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,NkT,NkN,Nz)
    EV=zeros(Float64,Nb,NkT,NkN,Nz)
    er=1.0/SDF_Lenders(1.0,0.0,par)
    qbar=(γ+(1.0-γ)*κ)/(er-(1.0-γ))
    q1=qbar*ones(Float64,Nb,NkT,NkN,Nz)
    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)
    #Policy functions
    kNprime_D=zeros(Float64,NkT,NkN,Nz)
    kTprime_D=zeros(Float64,NkT,NkN,Nz)
    kNprime=zeros(Float64,Nb,NkT,NkN,Nz)
    kTprime=zeros(Float64,Nb,NkT,NkN,Nz)
    bprime=zeros(Float64,Nb,NkT,NkN,Nz)
    Tr=zeros(Float64,Nb,NkT,NkN,Nz)
    yD=zeros(Float64,NkT,NkN,Nz)
    yNP=zeros(Float64,NkN,Nz)
    ccNP=zeros(Float64,NkN,Nz)
    yTP=zeros(Float64,NkT,Nz)
    cD=zeros(Float64,NkT,NkN,Nz)
    cP=zeros(Float64,Nb,NkT,NkN,Nz)
    itp_kNprime_D=CreateInterpolation_Policies(kNprime_D,true,GRIDS)
    itp_kTprime_D=CreateInterpolation_Policies(kTprime_D,true,GRIDS)
    itp_kNprime=CreateInterpolation_Policies(kNprime,false,GRIDS)
    itp_kTprime=CreateInterpolation_Policies(kTprime,false,GRIDS)
    itp_bprime=CreateInterpolation_Policies(bprime,false,GRIDS)
    itp_Tr=CreateInterpolation_Policies(Tr,false,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,kNprime_D,kTprime_D,kNprime,kTprime,bprime,Tr,yD,yNP,ccNP,yTP,cD,cP,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kNprime_D,itp_kTprime_D,itp_kNprime,itp_kTprime,itp_bprime,itp_Tr)
end

function SolutionEndOfTime(GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb, δ, φ, γ, κ, kNlowOpt, kNhighOpt, kTlowOpt, kThighOpt, cmin = par
    @unpack GR_z, GR_kN, GR_kT, GR_b = GRIDS
    SOLUTION=InitiateEmptySolution(GRIDS,par)
    er=1.0/SDF_Lenders(1.0,0.0,par)
    qbar=(γ+(1.0-γ)*κ)/(er-(1.0-γ))
    #Production of intermediates
    @unpack E_in, aN = par
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        for kN_ind in 1:NkN
            kN=GR_kN[kN_ind]
            SOLUTION.yNP[kN_ind,z_ind]=NonTradedProduction(z,kN,par)
            SOLUTION.ccNP[kN_ind,z_ind]=aN*(SOLUTION.yNP[kN_ind,z_ind]^E_in)
        end
        for kT_ind in 1:NkT
            kT=GR_kT[kT_ind]
            SOLUTION.yTP[kT_ind,z_ind]=TradedProduction(z,kT,par)
        end
    end
    #Value Functions in very last period
    #Assume they leave capital constant so that adjt cost is 0
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        zD=zDefault(z,par)
        for kN_ind in 1:NkN
            kN=GR_kN[kN_ind]
            #Will not use as policy here, only for next iteration starting value
            kNprime=kN
            for kT_ind in 1:NkT
                kT=GR_kT[kT_ind]
                #Will not use as policy here, only for next iteration starting value
                kTprime=kT
                SOLUTION.kNprime_D[kT_ind,kN_ind,z_ind]=kNprime
                SOLUTION.kTprime_D[kT_ind,kN_ind,z_ind]=kTprime
                SOLUTION.yD[kT_ind,kN_ind,z_ind]=FinalOutput(zD,kN,kT,0.0,par)
                # cD=ConsNet(SOLUTION.yD[kT_ind,kN_ind,z_ind],kN,kT,kNprime,kTprime,par)
                #Consume output minus depreciation of capital
                cD=SOLUTION.yD[kT_ind,kN_ind,z_ind]-δ*(kN+kT)
                SOLUTION.VD[kT_ind,kN_ind,z_ind]=Utility(cD,par)
                for b_ind in 1:Nb
                    #Will not use as policy here, only for next iteration starting value
                    SOLUTION.kNprime[b_ind,kT_ind,kN_ind,z_ind]=kNprime
                    SOLUTION.kTprime[b_ind,kT_ind,kN_ind,z_ind]=kTprime
                    b=GR_b[b_ind]
                    #Pay all outstanding debt obligations
                    T=-(γ+(1.0-γ)*(κ+qbar))*b
                    # T=-(γ+(1.0-γ)*κ)*b
                    SOLUTION.Tr[b_ind,kT_ind,kN_ind,z_ind]=T
                    yP=FinalOutput(z,kN,kT,T,par)
                    if yP>0.0
                        #Consume output minus depreciation of capital
                        cP=yP-δ*(kN+kT)
                        if cP>0.0
                            SOLUTION.VP[b_ind,kT_ind,kN_ind,z_ind]=Utility(cP,par)
                        else
                            SOLUTION.VP[b_ind,kT_ind,kN_ind,z_ind]=Utility(cmin,par)+T
                        end
                    else
                        SOLUTION.VP[b_ind,kT_ind,kN_ind,z_ind]=Utility(cmin,par)+T
                    end
                    if SOLUTION.VD[kT_ind,kN_ind,z_ind]>SOLUTION.VP[b_ind,kT_ind,kN_ind,z_ind]
                        SOLUTION.V[b_ind,kT_ind,kN_ind,z_ind]=SOLUTION.VD[kT_ind,kN_ind,z_ind]
                    else
                        SOLUTION.V[b_ind,kT_ind,kN_ind,z_ind]=SOLUTION.VP[b_ind,kT_ind,kN_ind,z_ind]
                    end
                end
            end
        end
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kNprime_D=CreateInterpolation_Policies(SOLUTION.kNprime_D,true,GRIDS)
    SOLUTION.itp_kTprime_D=CreateInterpolation_Policies(SOLUTION.kTprime_D,true,GRIDS)
    SOLUTION.itp_kNprime=CreateInterpolation_Policies(SOLUTION.kNprime,false,GRIDS)
    SOLUTION.itp_kTprime=CreateInterpolation_Policies(SOLUTION.kTprime,false,GRIDS)
    SOLUTION.itp_Tr=CreateInterpolation_Policies(SOLUTION.Tr,false,GRIDS)
    #Loop over all states to compute expectations over p and z
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        for kN_ind in 1:NkN
            kNprime=GR_kN[kN_ind]
            for kT_ind in 1:NkT
                kTprime=GR_kT[kT_ind]
                I=CartesianIndex(kT_ind,kN_ind,z_ind)
                Expectation_Default!(I,SOLUTION,GRIDS,par)
            end
        end
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        for kN_ind in 1:NkN
            kNprime=GR_kN[kN_ind]
            for kT_ind in 1:NkT
                kTprime=GR_kT[kT_ind]
                for b_ind in 1:Nb
                    bprime=GR_b[b_ind]
                    I=CartesianIndex(b_ind,kT_ind,kN_ind,z_ind)
                    Expectation_Repayment!(I,SOLUTION,GRIDS,par)
                end
            end
        end
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    return SOLUTION
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct States_TS
    z::Array{Float64,1}
    Def::Array{Float64,1}
    KN::Array{Float64,1}
    KT::Array{Float64,1}
    B::Array{Float64,1}
    Spreads::Array{Float64,1}
end

function Simulate_z_shocks(T::Int64,GRIDS::Grids,par::Pars)
    @unpack μ_z, ρ_z, dist_ϵz = par
    ϵz_TS=rand(dist_ϵz,T)
    z_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        if t==1
            z_TS[t]=μ_z
        else
            z_TS[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_TS[t-1])+ϵz_TS[t])
        end
    end
    return z_TS, ϵz_TS
end

function Simulate_z_shocks_z0(z0::Float64,T::Int64,GRIDS::Grids,par::Pars)
    @unpack μ_z, ρ_z, dist_ϵz = par
    ϵz_TS=rand(dist_ϵz,T)
    z_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        if t==1
            z_TS[t]=z0
        else
            z_TS[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_TS[t-1])+ϵz_TS[t])
        end
    end
    return z_TS, ϵz_TS
end

function ComputeSpreads(z::Float64,kNprime::Float64,kTprime::Float64,
                        bprime::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, γ, κ = par
    @unpack itp_q1 = SOLUTION
    q=max(itp_q1(bprime,kTprime,kNprime,z),1e-2)
    ib=((1.0-log(q/(γ+(1.0-γ)*(κ+q))))^4.0)-1.0
    rf=((1.0+r_star)^4.0)-1.0
    return 100.0*(ib-rf)
end

function SimulateStates_Long(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ = par
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
    @unpack itp_kNprime_D, itp_kTprime_D = SOLUTION
    #Simulate z, p, and oil discoveries
    z, ϵz=Simulate_z_shocks(Tsim,GRIDS,par)
    #Initiate vectors
    Def=Array{Float64,1}(undef,Tsim)
    KN=Array{Float64,1}(undef,Tsim)
    KT=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            KN[t], KT[t]=SteadyStateCapital(par)
            B[t]=0.0
            if itp_VD(KT[t],KN[t],z[t])<=itp_VP(B[t],KT[t],KN[t],z[t])
                Def[t]=0.0
                KN[t+1]=itp_kNprime(B[t],KT[t],KN[t],z[t])
                KT[t+1]=itp_kTprime(B[t],KT[t],KN[t],z[t])
                B[t+1]=max(itp_bprime(B[t],KT[t],KN[t],z[t]),0.0)
                Spreads[t]=ComputeSpreads(z[t],KN[t+1],KT[t+1],B[t+1],SOLUTION,GRIDS,par)
            else
                Def[t]=1.0
                KN[t+1]=itp_kNprime_D(KT[t],KN[t],z[t])
                KT[t+1]=itp_kTprime_D(KT[t],KN[t],z[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Tsim
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KT[t],KN[t],z[t])<=itp_VP(B[t],KT[t],KN[t],z[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KT[t],KN[t],z[t])<=itp_VP(B[t],KT[t],KN[t],z[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KT[t],KN[t],z[t])<=itp_VP(B[t],KT[t],KN[t],z[t])
                            Def[t]=0.0
                            KN[t+1]=itp_kNprime(B[t],KT[t],KN[t],z[t])
                            KT[t+1]=itp_kTprime(B[t],KT[t],KN[t],z[t])
                            B[t+1]=max(itp_bprime(B[t],KT[t],KN[t],z[t]),0.0)
                            Spreads[t]=ComputeSpreads(z[t],KN[t+1],KT[t+1],B[t+1],SOLUTION,GRIDS,par)
                        else
                            Def[t]=1.0
                            KN[t+1]=itp_kNprime_D(KT[t],KN[t],z[t])
                            KT[t+1]=itp_kTprime_D(KT[t],KN[t],z[t])
                            B[t+1]=0.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        KN[t+1]=itp_kNprime_D(KT[t],KN[t],z[t])
                        KT[t+1]=itp_kTprime_D(KT[t],KN[t],z[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KT[t],KN[t],z[t])<=itp_VP(B[t],KT[t],KN[t],z[t])
                        Def[t]=0.0
                        KN[t+1]=itp_kNprime(B[t],KT[t],KN[t],z[t])
                        KT[t+1]=itp_kTprime(B[t],KT[t],KN[t],z[t])
                        B[t+1]=max(itp_bprime(B[t],KT[t],KN[t],z[t]),0.0)
                        Spreads[t]=ComputeSpreads(z[t],KN[t+1],KT[t+1],B[t+1],SOLUTION,GRIDS,par)
                    else
                        Def[t]=1.0
                        KN[t+1]=itp_kNprime_D(KT[t],KN[t],z[t])
                        KT[t+1]=itp_kTprime_D(KT[t],KN[t],z[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            end
        end
    end
    return States_TS(z,Def,KN,KT,B,Spreads)
end

########## Compute moments
@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    MeanSpreads::Float64 = 0.0
    StdSpreads::Float64 = 0.0
    #Stocks
    Debt_GDP::Float64 = 0.0
    kN_GDP::Float64 = 0.0
    kT_GDP::Float64 = 0.0
    AvλK::Float64 = 0.0
    stdλK::Float64 = 0.0
    #Volatilities
    σ_GDP::Float64 = 0.0
    σ_con::Float64 = 0.0
    σ_inv::Float64 = 0.0
    #Cyclicality
    Corr_con_GDP::Float64 = 0.0
    Corr_inv_GDP::Float64 = 0.0
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_CA_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
end
###### Select moments of length Tmom that start in good standing and have been
###### in good standing after, at least, TsinceDefault=25 periods, consider no
###### discoveries in the sample for the moments

function GetOnePathForMoments(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, Tmom, TsinceDefault, Tsim = par
    STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
    #Start after drp, start at good standing Def[t0]=0
    t0=drp+1
    if STATES_TS.Def[t0]==1.0
        #Keep going until good standing
        while true
            t0=t0+1
            if STATES_TS.Def[t0]==0.0
                break
            end
        end
    end
    #Count TsinceDefault periods without default
    #Try at most 5 long samples
    NLongSamples=0
    tsince=0
    while true
        if t0+Tmom+10==Tsim
            NLongSamples=NLongSamples+1
            if NLongSamples<=5
                #Get another long sample
                STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
                t0=drp+1
                tsince=0
            else
                # println("failed to get appropriate moment sample")
                break
            end
        end
        if STATES_TS.Def[t0]==1.0
            tsince=0
        else
            tsince=tsince+1
        end
        t0=t0+1
        if STATES_TS.Def[t0]==0.0 && tsince>=TsinceDefault
            break
        end
    end
    t1=t0+Tmom-1
    return States_TS(STATES_TS.z[t0:t1],STATES_TS.Def[t0:t1],STATES_TS.KN[t0:t1],STATES_TS.KT[t0:t1],STATES_TS.B[t0:t1],STATES_TS.Spreads[t0:t1])
end

function hp_filter(y::Vector{Float64}, lambda::Float64)
    #Function from QuantEcon
    #Returns trend component
    n = length(y)
    @assert n >= 4

    diag2 = lambda*ones(n-2)
    diag1 = [ -2lambda; -4lambda*ones(n-3); -2lambda ]
    diag0 = [ 1+lambda; 1+5lambda; (1+6lambda)*ones(n-4); 1+5lambda; 1+lambda ]

    #D = spdiagm((diag2, diag1, diag0, diag1, diag2), (-2,-1,0,1,2))
    D = spdiagm(-2 => diag2, -1 => diag1, 0 => diag0, 1 => diag1, 2 => diag2)

    D\y
end

function ComputeMomentsOnce(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tmom, γ, κ, HPFilter_Par, δ = par
    @unpack itp_q1, itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
    @unpack itp_kNprime_D, itp_kTprime_D = SOLUTION
    STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
    #Compute easy moments
    DefaultPr=100.0*mean(STATES_TS.Def)
    MeanSpreads=sum((STATES_TS.Spreads) .* (STATES_TS.Def .== 0.0))/sum(STATES_TS.Def .== 0.0)
    StdSpreads=std(STATES_TS.Spreads)
    AvλK=mean(STATES_TS.KT ./ (STATES_TS.KT .+ STATES_TS.KN))
    stdλK=std(STATES_TS.KT ./ (STATES_TS.KT .+ STATES_TS.KN))
    #Compute other variables
    P_TS=Array{Float64,1}(undef,Tmom)
    Y_TS=Array{Float64,1}(undef,Tmom)
    GDP_TS=Array{Float64,1}(undef,Tmom)
    con_TS=Array{Float64,1}(undef,Tmom)
    inv_TS=Array{Float64,1}(undef,Tmom)
    TB_TS=Array{Float64,1}(undef,Tmom)
    CA_TS=Array{Float64,1}(undef,Tmom)
    while true
        # STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
        #Compute easy moments
        DefaultPr=100.0*mean(STATES_TS.Def)
        MeanSpreads=mean(STATES_TS.Spreads)
        StdSpreads=std(STATES_TS.Spreads)
        for t in 1:Tmom
            z=STATES_TS.z[t]
            kN=STATES_TS.KN[t]
            kT=STATES_TS.KT[t]
            b=STATES_TS.B[t]
            if t<Tmom
                kNprime=STATES_TS.KN[t+1]
                kTprime=STATES_TS.KT[t+1]
                bprime=STATES_TS.B[t+1]
            else
                if STATES_TS.Def[t]==0.0
                    kNprime=itp_kNprime(b,kT,kN,z)
                    kTprime=itp_kTprime(b,kT,kN,z)
                    bprime=itp_bprime(b,kT,kN,z)
                else
                    kNprime=itp_kNprime_D(kT,kN,z)
                    kTprime=itp_kTprime_D(kT,kN,z)
                    bprime=0.0
                end
            end
            if STATES_TS.Def[t]==0.0
                q=itp_q1(bprime,kTprime,kNprime,z)
                X=q*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
                y=FinalOutput(z,kN,kT,X,par)
                P=PriceFinalGood(z,kN,kT,X,par)
            else
                zD=zDefault(z,par)
                X=0.0
                y=FinalOutput(zD,kN,kT,X,par)
                P=PriceFinalGood(zD,kN,kT,X,par)
            end
            P_TS[t]=P
            Y_TS[t]=y
            GDP_TS[t]=P*y-X
            AdjCost=CapitalAdjustment(kNprime,kN,par)+CapitalAdjustment(kTprime,kT,par)
            inv_TS[t]=P*((kNprime+kTprime)-(1.0-δ)*(kN+kT))
            con_TS[t]=GDP_TS[t]-inv_TS[t]+X-P*AdjCost
            TB_TS[t]=-X
            CA_TS[t]=-(bprime-b)
        end
        #reject samples with negative consumption
        if minimum(abs.(con_TS))>=0.0
            break
        else
            STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
        end
    end
    Debt_GDP=sum((100 .* (STATES_TS.B ./ (1.0 .* GDP_TS))) .* (STATES_TS.Def .== 0.0))/sum(STATES_TS.Def .== 0.0)
    kN_GDP=sum(((STATES_TS.KN) ./ (Y_TS)) .* (STATES_TS.Def .== 0.0))/sum(STATES_TS.Def .== 0.0)
    kT_GDP=sum(((STATES_TS.KT) ./ (Y_TS)) .* (STATES_TS.Def .== 0.0))/sum(STATES_TS.Def .== 0.0)
    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(GDP_TS))
    GDP_trend=hp_filter(log_GDP,HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    #Investment
    log_inv=log.(abs.(inv_TS))
    inv_trend=hp_filter(log_inv,HPFilter_Par)
    inv_cyc=100.0*(log_inv .- inv_trend)
    #Consumption
    log_con=log.(abs.(con_TS))
    con_trend=hp_filter(log_con,HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Volatilities
    σ_GDP=std(GDP_cyc)
    σ_con=std(con_cyc)
    σ_inv=std(inv_cyc)
    #Correlations with GDP
    Corr_con_GDP=cor(GDP_cyc,con_cyc)
    Corr_inv_GDP=cor(GDP_cyc,inv_cyc)
    Corr_Spreads_GDP=cor(GDP_cyc,STATES_TS.Spreads)
    Corr_CA_GDP=cor(GDP_cyc,100.0 .* (CA_TS ./ GDP_TS))
    Corr_TB_GDP=cor(GDP_cyc,100.0 .* (TB_TS ./ GDP_TS))
    return Moments(DefaultPr,MeanSpreads,StdSpreads,Debt_GDP,kN_GDP,kT_GDP,AvλK,stdλK,σ_GDP,σ_con,σ_inv,Corr_con_GDP,Corr_inv_GDP,Corr_Spreads_GDP,Corr_CA_GDP,Corr_TB_GDP)
end

function AverageMomentsManySamples(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    Random.seed!(1234)
    #Initiate them at 0.0 to facilitate average across samples
    MOMENTS=Moments()
    for i in 1:NSamplesMoments
        # println("Sample $i for moments")
        MOMS=ComputeMomentsOnce(SOLUTION,GRIDS,par)
        #Default, spreads, and Debt
        MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        #Stocks
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        MOMENTS.kN_GDP=MOMENTS.kN_GDP+MOMS.kN_GDP/NSamplesMoments
        MOMENTS.kT_GDP=MOMENTS.kT_GDP+MOMS.kT_GDP/NSamplesMoments
        MOMENTS.AvλK=MOMENTS.AvλK+MOMS.AvλK/NSamplesMoments
        MOMENTS.stdλK=MOMENTS.stdλK+MOMS.stdλK/NSamplesMoments
        #Volatilities
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
        MOMENTS.σ_inv=MOMENTS.σ_inv+MOMS.σ_inv/NSamplesMoments
        #Cyclicality
        MOMENTS.Corr_con_GDP=MOMENTS.Corr_con_GDP+MOMS.Corr_con_GDP/NSamplesMoments
        MOMENTS.Corr_inv_GDP=MOMENTS.Corr_inv_GDP+MOMS.Corr_inv_GDP/NSamplesMoments
        MOMENTS.Corr_Spreads_GDP=MOMENTS.Corr_Spreads_GDP+MOMS.Corr_Spreads_GDP/NSamplesMoments
        MOMENTS.Corr_CA_GDP=MOMENTS.Corr_CA_GDP+MOMS.Corr_CA_GDP/NSamplesMoments
        MOMENTS.Corr_TB_GDP=MOMENTS.Corr_TB_GDP+MOMS.Corr_TB_GDP/NSamplesMoments
    end
    return MOMENTS
end

function ComputeAndSaveMoments(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    MOMENTS=AverageMomentsManySamples(SOLUTION,GRIDS,par)
    RowNames=["DefaultPr";
              "MeanSpreads";
              "StdSpreads";
              "Debt_GDP";
              "kN_GDP";
              "kT_GDP";
              "AvλK";
              "stdλK";
              "σ_GDP";
              "σ_con";
              "σ_inv";
              "Corr_con_GDP";
              "Corr_inv_GDP";
              "Corr_Spreads_GDP";
              "Corr_CA_GDP";
              "Corr_TB_GDP"]
    Values=[MOMENTS.DefaultPr;
            MOMENTS.MeanSpreads;
            MOMENTS.StdSpreads;
            MOMENTS.Debt_GDP;
            MOMENTS.kN_GDP;
            MOMENTS.kT_GDP;
            MOMENTS.AvλK;
            MOMENTS.stdλK;
            MOMENTS.σ_GDP;
            MOMENTS.σ_con;
            MOMENTS.σ_inv;
            MOMENTS.Corr_con_GDP;
            MOMENTS.Corr_inv_GDP;
            MOMENTS.Corr_Spreads_GDP;
            MOMENTS.Corr_CA_GDP;
            MOMENTS.Corr_TB_GDP]
    MAT=[RowNames Values]
    writedlm("Moments.csv",MAT,',')
end

################################################################################
### Functions to solve and save solution in CSV
################################################################################
function SaveSolution(SOLUTION::Solution)
    #Save vectors of repayment
    @unpack VP, V, EV, kNprime, kTprime, bprime, q1, Tr = SOLUTION
    MAT=reshape(VP,(:))
    MAT=hcat(MAT,reshape(V,(:)))
    MAT=hcat(MAT,reshape(EV,(:)))
    MAT=hcat(MAT,reshape(kNprime,(:)))
    MAT=hcat(MAT,reshape(kTprime,(:)))
    MAT=hcat(MAT,reshape(bprime,(:)))
    MAT=hcat(MAT,reshape(q1,(:)))
    MAT=hcat(MAT,reshape(Tr,(:)))
    writedlm("Repayment.csv",MAT,',')
    #Save vectors of default
    @unpack VD, EVD, kNprime_D, kTprime_D = SOLUTION
    MAT=reshape(VD,(:))
    MAT=hcat(MAT,reshape(EVD,(:)))
    MAT=hcat(MAT,reshape(kNprime_D,(:)))
    MAT=hcat(MAT,reshape(kTprime_D,(:)))
    writedlm("Default.csv",MAT,',')
    return nothing
end

function Unpack_Solution(FOLDER::String,GRIDS::Grids,par::Pars)
    #The files Repayment.csv and Default.csv must be in FOLDER
    #for this function to work
    @unpack Nz, NkN, NkT, Nb = par
    #Unpack Matrices with data
    if FOLDER==" "
        MAT_R=readdlm("Repayment.csv",',')
        MAT_D=readdlm("Default.csv",',')
    else
        MAT_R=readdlm("$FOLDER\\Repayment.csv",',')
        MAT_D=readdlm("$FOLDER\\Default.csv",',')
    end
    #Initiate empty solution
    SOL=InitiateEmptySolution(GRIDS,par)
    #Allocate vectors into matrices
    #Repayment
    I=(Nb,NkT,NkN,Nz)
    SOL.VP=reshape(MAT_R[:,1],I)
    SOL.V=reshape(MAT_R[:,2],I)
    SOL.EV=reshape(MAT_R[:,3],I)
    SOL.kNprime=reshape(MAT_R[:,4],I)
    SOL.kTprime=reshape(MAT_R[:,5],I)
    SOL.bprime=reshape(MAT_R[:,6],I)
    SOL.q1=reshape(MAT_R[:,7],I)
    SOL.Tr=reshape(MAT_R[:,8],I)
    #Default
    I=(NkT,NkN,Nz)
    SOL.VD=reshape(MAT_D[:,1],I)
    SOL.EVD=reshape(MAT_D[:,2],I)
    SOL.kNprime_D=reshape(MAT_D[:,3],I)
    SOL.kTprime_D=reshape(MAT_D[:,4],I)
    #Create interpolation objects
    SOL.itp_VD=CreateInterpolation_ValueFunctions(SOL.VD,true,GRIDS)
    SOL.itp_VP=CreateInterpolation_ValueFunctions(SOL.VP,false,GRIDS)
    SOL.itp_V=CreateInterpolation_ValueFunctions(SOL.V,false,GRIDS)
    SOL.itp_EVD=CreateInterpolation_ValueFunctions(SOL.EVD,true,GRIDS)
    SOL.itp_EV=CreateInterpolation_ValueFunctions(SOL.EV,false,GRIDS)
    SOL.itp_q1=CreateInterpolation_Price(SOL.q1,GRIDS)
    SOL.itp_kNprime_D=CreateInterpolation_Policies(SOL.kNprime_D,true,GRIDS)
    SOL.itp_kTprime_D=CreateInterpolation_Policies(SOL.kTprime_D,true,GRIDS)
    SOL.itp_kNprime=CreateInterpolation_Policies(SOL.kNprime,false,GRIDS)
    SOL.itp_kTprime=CreateInterpolation_Policies(SOL.kTprime,false,GRIDS)
    SOL.itp_bprime=CreateInterpolation_Policies(SOL.bprime,false,GRIDS)
    SOL.itp_Tr=CreateInterpolation_Policies(SOL.Tr,false,GRIDS)
    return SOL
end

function SolveAndSaveModel_VFI(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTol, Tolpct_q, cnt_max = par
    println("Preparing solution guess")
    SOLUTION_CURRENT=SolutionEndOfTime(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    println("Starting VFI")
    while ((dst_V>Tol_V && rdts_V>relTol) || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution!(SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        rdts_V=max(rdst_D,rdst_P)
        dst_V=rdst_D
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        println("cnt=$cnt, rdst_D=$rdst_D%, rdst_P=$rdst_P%, dst_q=$dst_q")
        println("    $cnt,  dst_D=$dst_D ,  dst_P=$dst_P ,       $NotConvPct% of q not converged")
    end
    SaveSolution(SOLUTION_NEXT)
    println("Compute Moments")
    ComputeAndSaveMoments(SOLUTION_NEXT,GRIDS,par)
    return nothing
end

function SolveModel_VFI(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTol, Tolpct_q, cnt_max = par
    println("Preparing solution guess")
    SOLUTION_CURRENT=SolutionEndOfTime(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    println("Starting VFI")
    while ((dst_V>Tol_V && rdts_V>relTol) || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution!(SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        rdts_V=max(rdst_D,rdst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        # println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P, dst_q=$dst_q, $NotConvPct% of q not converged")
        println("cnt=$cnt, rdst_D=$rdst_D%, rdst_P=$rdst_P%, dst_q=$dst_q")
        println("    $cnt,  dst_D=$dst_D ,  dst_P=$dst_P ,       $NotConvPct% of q not converged")
    end
    return SOLUTION_NEXT
end

###############################################################################
#Functions to Solve decentralized model
###############################################################################

#Prices for households's FOCs
function rN_dec(z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack αN, A = par
    pN=PriceNonTraded(z,kN,kT,T,par)
    return αN*pN*z*A*(kN^(αN-1.0))
end

function rT_dec(z::Float64,kT::Float64,par::Pars)
    @unpack αT = par
    return αT*z*A*(kT^(αT-1.0))
end

function RN_dec(kNprime::Float64,z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack δ = par
    rN=rN_dec(z,kN,kT,T,par)
    P=PriceFinalGood(z,kN,kT,T,par)
    PkN=1.0+dΨ_dkprime(kNprime,kN,par)
    return (rN/P)+(1.0-δ)*PkN-dΨ_dk(kNprime,kN,par)
end

function RT_dec(kTprime::Float64,z::Float64,kN::Float64,kT::Float64,T::Float64,par::Pars)
    @unpack δ = par
    rT=rT_dec(z,kT,par)
    P=PriceFinalGood(z,kN,kT,T,par)
    PkT=1.0+dΨ_dkprime(kTprime,kT,par)
    return (rT/P)+(1.0-δ)*PkT-dΨ_dk(kTprime,kT,par)
end

#Interpolation objects for households' FOCs
@with_kw mutable struct HH_itpObjects{T1,T2,T3,T4}
    #Arrays
    cDef::T1
    cRep::T2
    RN_Def::T1
    RT_Def::T1
    RN_Rep::T2
    RT_Rep::T2
    #Interpolation objects
    itp_cDef::T3
    itp_cRep::T4
    itp_RN_Def::T3
    itp_RT_Def::T3
    itp_RN_Rep::T4
    itp_RT_Rep::T4
end

function UpdateConsumptionPolicy!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin = par
    @unpack GR_z, GR_kN, GR_kT, GR_b = GRIDS
    @unpack kNprime_D, kTprime_D = SOLUTION
    @unpack kNprime, kTprime, bprime, Tr = SOLUTION
    #Loop over all states to fill consumption in default
    for I in CartesianIndices(HH_OBJ.cDef)
        (kT_ind,kN_ind,z_ind)=Tuple(I)
        kNprimef=kNprime_D[I]
        kTprimef=kTprime_D[I]
        z=GR_z[z_ind]
        zD=zDefault(z,par)
        kN=GR_kN[kN_ind]
        kT=GR_kT[kT_ind]
        T=0.0
        y=FinalOutput(zD,kN,kT,T,par)
        if y>0.0
            HH_OBJ.cDef[I]=max(ConsNet(y,kN,kT,kNprimef,kTprimef,par),cmin)
        else
            HH_OBJ.cDef[I]=cmin
        end
    end
    HH_OBJ.itp_cDef=CreateInterpolation_cPol(HH_OBJ.cDef,true,GRIDS)
    #Loop over all states to fill consumption in repayment
    for I in CartesianIndices(HH_OBJ.cRep)
        (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
        kNprimef=kNprime[I]
        kTprimef=kTprime[I]
        z=GR_z[z_ind]
        kN=GR_kN[kN_ind]
        kT=GR_kT[kT_ind]
        T=Tr[I]
        y=FinalOutput(z,kN,kT,T,par)
        if y>0.0
            HH_OBJ.cRep[I]=max(ConsNet(y,kN,kT,kNprimef,kTprimef,par),cmin)
        else
            HH_OBJ.cRep[I]=cmin
        end
    end
    HH_OBJ.itp_cRep=CreateInterpolation_cPol(HH_OBJ.cRep,false,GRIDS)
    return nothing
end

function UpdateCapitalReturns!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb, cmin = par
    @unpack GR_z, GR_kN, GR_kT, GR_b = GRIDS
    @unpack kNprime_D, kTprime_D = SOLUTION
    @unpack kNprime, kTprime, Tr = SOLUTION
    ######Do default first
    #Loop over all states to fill Ri in default
    for I in CartesianIndices(HH_OBJ.RN_Def)
        (kT_ind,kN_ind,z_ind)=Tuple(I)
        kNprimef=kNprime_D[I]
        kTprimef=kTprime_D[I]
        z=GR_z[z_ind]
        zD=zDefault(z,par)
        kN=GR_kN[kN_ind]
        kT=GR_kT[kT_ind]
        T=0.0
        HH_OBJ.RN_Def[I]=RN_dec(kNprimef,zD,kN,kT,T,par)
        HH_OBJ.RT_Def[I]=RT_dec(kTprimef,zD,kN,kT,T,par)
    end
    HH_OBJ.itp_RN_Def=CreateInterpolation_cPol(HH_OBJ.RN_Def,true,GRIDS)
    HH_OBJ.itp_RT_Def=CreateInterpolation_cPol(HH_OBJ.RT_Def,true,GRIDS)

    ######Now do repayment
    #Loop over all states to fill Ri in default
    for I in CartesianIndices(HH_OBJ.RN_Rep)
        (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
        kNprimef=kNprime[I]
        kTprimef=kTprime[I]
        z=GR_z[z_ind]
        kN=GR_kN[kN_ind]
        kT=GR_kT[kT_ind]
        T=Tr[I]
        HH_OBJ.RN_Rep[I]=RN_dec(kNprimef,z,kN,kT,T,par)
        HH_OBJ.RT_Rep[I]=RT_dec(kTprimef,z,kN,kT,T,par)
    end
    HH_OBJ.itp_RN_Rep=CreateInterpolation_cPol(HH_OBJ.RN_Rep,false,GRIDS)
    HH_OBJ.itp_RT_Rep=CreateInterpolation_cPol(HH_OBJ.RT_Rep,false,GRIDS)
    return nothing
end

#Functions to compute integrands in FOCs given (s,x) and a try of K'
function Compute_sdf_HH(c::Float64,Defprime::Int64,zprime::Float64,
                        kNprime::Float64,kTprime::Float64,bprime::Float64,
                        HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, σ, cmin = par
    @unpack itp_cDef, itp_cRep = HH_OBJ
    if Defprime==1
        cprime=max(cmin,itp_cDef(kTprime,kNprime,zprime))
    else
        cprime=max(cmin,itp_cRep(bprime,kTprime,kNprime,zprime))
    end
    return β*((c/cprime)^σ)
end

function integrand_HH_FOC_N_Def(c::Float64,zprime::Float64,kNprime::Float64,kTprime::Float64,
                        HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack θ = par
    @unpack itp_RN_Def, itp_RN_Rep = HH_OBJ
    sdfDD=Compute_sdf_HH(c,1,zprime,kNprime,kTprime,0.0,HH_OBJ,SOLUTION,GRIDS,par)
    sdfDP=Compute_sdf_HH(c,0,zprime,kNprime,kTprime,0.0,HH_OBJ,SOLUTION,GRIDS,par)
    RD=itp_RN_Def(kTprime,kNprime,zprime)
    RP=itp_RN_Rep(0.0,kTprime,kNprime,zprime)
    return θ*(sdfDP*RP)+(1.0-θ)*(sdfDD*RD)
end

function integrand_HH_FOC_T_Def(c::Float64,zprime::Float64,kNprime::Float64,kTprime::Float64,
                        HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack θ = par
    @unpack itp_RT_Def, itp_RT_Rep = HH_OBJ
    sdfDD=Compute_sdf_HH(c,1,zprime,kNprime,kTprime,0.0,HH_OBJ,SOLUTION,GRIDS,par)
    sdfDP=Compute_sdf_HH(c,0,zprime,kNprime,kTprime,0.0,HH_OBJ,SOLUTION,GRIDS,par)
    RD=itp_RT_Def(kTprime,kNprime,zprime)
    RP=itp_RT_Rep(0.0,kTprime,kNprime,zprime)
    return θ*(sdfDP*RP)+(1.0-θ)*(sdfDD*RD)
end

function integrand_HH_FOC_N_Rep(c::Float64,zprime::Float64,kNprime::Float64,kTprime::Float64,bprime::Float64,
                        HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_VD, itp_VP = SOLUTION
    if itp_VD(kTprime,kNprime,zprime)>itp_VP(bprime,kTprime,kNprime,zprime)
        @unpack itp_RN_Def = HH_OBJ
        sdfPD=Compute_sdf_HH(c,1,zprime,kNprime,kTprime,0.0,HH_OBJ,SOLUTION,GRIDS,par)
        RD=itp_RN_Def(kTprime,kNprime,zprime)
        return sdfPD*RD
    else
        @unpack itp_RN_Rep = HH_OBJ
        sdfPP=Compute_sdf_HH(c,0,zprime,kNprime,kTprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
        RP=itp_RN_Rep(bprime,kTprime,kNprime,zprime)
        return sdfPP*RP
    end
end

function integrand_HH_FOC_T_Rep(c::Float64,zprime::Float64,kNprime::Float64,kTprime::Float64,bprime::Float64,
                        HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_VD, itp_VP, itp_Tr = SOLUTION
    if itp_VD(kTprime,kNprime,zprime)>itp_VP(bprime,kTprime,kNprime,zprime)
        @unpack itp_RT_Def = HH_OBJ
        sdfPD=Compute_sdf_HH(c,1,zprime,kNprime,kTprime,0.0,HH_OBJ,SOLUTION,GRIDS,par)
        RD=itp_RT_Def(kTprime,kNprime,zprime)
        return sdfPD*RD
    else
        @unpack itp_RT_Rep = HH_OBJ
        sdfPP=Compute_sdf_HH(c,0,zprime,kNprime,kTprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
        RP=itp_RT_Rep(bprime,kTprime,kNprime,zprime)
        return sdfPP*RP
    end
end

#Update household objects
function UpdateHH_Obj!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateConsumptionPolicy!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateCapitalReturns!(HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

function InitiateHH_Obj(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb, cmin, δ = par
    @unpack GR_z, GR_kN, GR_kT, GR_b = GRIDS
    @unpack kNprime, kTprime, bprime, kNprime_D, kTprime_D, Tr = SOLUTION
    #Allocate arrays for end of time
    cDef=zeros(Float64,NkT,NkN,Nz)
    cRep=zeros(Float64,Nb,NkT,NkN,Nz)
    RN_Def=zeros(Float64,NkT,NkN,Nz)
    RT_Def=zeros(Float64,NkT,NkN,Nz)
    RN_Rep=zeros(Float64,Nb,NkT,NkN,Nz)
    RT_Rep=zeros(Float64,Nb,NkT,NkN,Nz)
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        zD=zDefault(z,par)
        for kN_ind in 1:NkN
            kN=GR_kN[kN_ind]
            for kT_ind in 1:NkT
                kT=GR_kT[kT_ind]
                T=0.0
                y=FinalOutput(zD,kN,kT,T,par)
                kNprimef=kNprime_D[kT_ind,kN_ind,z_ind]
                kTprimef=kTprime_D[kT_ind,kN_ind,z_ind]
                if y>0.0
                    cDef[kT_ind,kN_ind,z_ind]=max(ConsNet(y,kN,kT,kNprimef,kTprimef,par),cmin)
                else
                    cDef[kT_ind,kN_ind,z_ind]=cmin
                end
                RN_Def[kT_ind,kN_ind,z_ind]=RN_dec(kNprimef,zD,kN,kT,T,par)
                RT_Def[kT_ind,kN_ind,z_ind]=RT_dec(kTprimef,zD,kN,kT,T,par)
                for b_ind in 1:Nb
                    kNprimef=kNprime[b_ind,kT_ind,kN_ind,z_ind]
                    kTprimef=kTprime[b_ind,kT_ind,kN_ind,z_ind]
                    T=Tr[b_ind,kT_ind,kN_ind,z_ind]
                    y=FinalOutput(z,kN,kT,T,par)
                    if y>0.0
                        cRep[b_ind,kT_ind,kN_ind,z_ind]=max(ConsNet(y,kN,kT,kNprimef,kTprimef,par),cmin)
                    else
                        cRep[b_ind,kT_ind,kN_ind,z_ind]=cmin
                    end
                    RN_Rep[b_ind,kT_ind,kN_ind,z_ind]=RN_dec(kNprimef,z,kN,kT,T,par)
                    RT_Rep[b_ind,kT_ind,kN_ind,z_ind]=RT_dec(kTprimef,z,kN,kT,T,par)
                end
            end
        end
    end
    #Create interpolation objects
    itp_cDef=CreateInterpolation_cPol(cDef,true,GRIDS)
    itp_cRep=CreateInterpolation_cPol(cRep,false,GRIDS)
    itp_RN_Def=CreateInterpolation_cPol(RN_Def,true,GRIDS)
    itp_RT_Def=CreateInterpolation_cPol(RT_Def,true,GRIDS)
    itp_RN_Rep=CreateInterpolation_cPol(RN_Rep,false,GRIDS)
    itp_RT_Rep=CreateInterpolation_cPol(RT_Rep,false,GRIDS)
    #Arrange objects in structure
    HH_OBJ=HH_itpObjects(cDef,cRep,RN_Def,RT_Def,RN_Rep,RT_Rep,itp_cDef,itp_cRep,itp_RN_Def,itp_RT_Def,itp_RN_Rep,itp_RT_Rep)
    return HH_OBJ
end

#Functions to compute FOCs given (s,x) and a try of K'
function HH_FOC_N_Def(z_ind::Int64,kN::Float64,kT::Float64,
                      kNprime::Float64,kTprime::Float64,HH_OBJ::HH_itpObjects,
                      SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin = par
    @unpack ϵz_weights, ZPRIME, PDFz, GR_z = GRIDS
    #Compute present consumption
    z=GR_z[z_ind]
    zD=zDefault(z,par)
    T=0.0
    y=FinalOutput(zD,kN,kT,T,par)
    c=max(cmin,ConsNet(y,kN,kT,kNprime,kTprime,par))
    #compute expectation over z
    foo(zprime::Float64)=integrand_HH_FOC_N_Def(c,zprime,kNprime,kTprime,HH_OBJ,SOLUTION,GRIDS,par)
    Ev=dot(ϵz_weights,PDFz[z_ind,:] .* foo.(ZPRIME[z_ind,:]))
    #compute extra term and return FOC
    PkN=1.0+dΨ_dkprime(kNprime,kN,par)
    return Ev-PkN
end

function HH_FOC_T_Def(z_ind::Int64,kN::Float64,kT::Float64,
                      kNprime::Float64,kTprime::Float64,HH_OBJ::HH_itpObjects,
                      SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack cmin = par
    @unpack ϵz_weights, ZPRIME, PDFz, GR_z = GRIDS
    #Compute present consumption
    z=GR_z[z_ind]
    zD=zDefault(z,par)
    T=0.0
    y=FinalOutput(zD,kN,kT,T,par)
    c=max(cmin,ConsNet(y,kN,kT,kNprime,kTprime,par))
    #compute expectation over z
    foo(zprime::Float64)=integrand_HH_FOC_T_Def(c,zprime,kNprime,kTprime,HH_OBJ,SOLUTION,GRIDS,par)
    Ev=dot(ϵz_weights,PDFz[z_ind,:] .* foo.(ZPRIME[z_ind,:]))
    #compute extra term and return FOC
    PkT=1.0+dΨ_dkprime(kTprime,kT,par)
    return Ev-PkT
end

function HH_FOC_N_Rep(z_ind::Int64,kN::Float64,kT::Float64,b::Float64,
                      kNprime::Float64,kTprime::Float64,bprime::Float64,
                      HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ, cmin = par
    @unpack ϵz_weights, ZPRIME, PDFz, GR_z = GRIDS
    @unpack itp_q1 = SOLUTION
    #Compute present consumption
    z=GR_z[z_ind]
    qq=itp_q1(bprime,kTprime,kNprime,z)
    T=qq*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
    y=FinalOutput(z,kN,kT,T,par)
    c=max(cmin,ConsNet(y,kN,kT,kNprime,kTprime,par))
    #compute expectation over z
    foo(zprime::Float64)=integrand_HH_FOC_N_Rep(c,zprime,kNprime,kTprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    Ev=dot(ϵz_weights,PDFz[z_ind,:] .* foo.(ZPRIME[z_ind,:]))
    #compute extra term and return FOC
    PkN=1.0+dΨ_dkprime(kNprime,kN,par)
    return Ev-PkN
end

function HH_FOC_T_Rep(z_ind::Int64,kN::Float64,kT::Float64,b::Float64,
                      kNprime::Float64,kTprime::Float64,bprime::Float64,
                      HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ, cmin = par
    @unpack ϵz_weights, ZPRIME, PDFz, GR_z = GRIDS
    @unpack itp_q1 = SOLUTION
    #Compute present consumption
    z=GR_z[z_ind]
    qq=itp_q1(bprime,kTprime,kNprime,z)
    T=qq*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
    y=FinalOutput(z,kN,kT,T,par)
    c=max(cmin,ConsNet(y,kN,kT,kNprime,kTprime,par))
    #compute expectation over z
    foo(zprime::Float64)=integrand_HH_FOC_T_Rep(c,zprime,kNprime,kTprime,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    Ev=dot(ϵz_weights,PDFz[z_ind,:] .* foo.(ZPRIME[z_ind,:]))
    #compute extra term and return FOC
    PkT=1.0+dΨ_dkprime(kTprime,kT,par)
    return Ev-PkT
end

#Functions to compute optimal capital policy from FOCs and previous iteration
function F_and_Jacobian_Def!(x::Array{Float64,1},F::Array{Float64,1},J::Array{Float64,2},
                             z_ind::Int64,kN::Float64,kT::Float64,HH_OBJ::HH_itpObjects,
                             SOLUTION::Solution,GRIDS::Grids,par::Pars)
    hh=1e-6
    #Update F
    F[1]=HH_FOC_N_Def(z_ind,kN,kT,x[1],x[2],HH_OBJ,SOLUTION,GRIDS,par)
    F[2]=HH_FOC_T_Def(z_ind,kN,kT,x[1],x[2],HH_OBJ,SOLUTION,GRIDS,par)

    #Update J
    #dF1/dkN
    fl=HH_FOC_N_Def(z_ind,kN,kT,x[1]-hh,x[2],HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_N_Def(z_ind,kN,kT,x[1]+hh,x[2],HH_OBJ,SOLUTION,GRIDS,par)
    J[1,1]=(fh-fl)/(2.0*hh)
    #dF1/dkT
    fl=HH_FOC_N_Def(z_ind,kN,kT,x[1],x[2]-hh,HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_N_Def(z_ind,kN,kT,x[1],x[2]+hh,HH_OBJ,SOLUTION,GRIDS,par)
    J[1,2]=(fh-fl)/(2.0*hh)

    #dF1/dkN
    fl=HH_FOC_T_Def(z_ind,kN,kT,x[1]-hh,x[2],HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_T_Def(z_ind,kN,kT,x[1]+hh,x[2],HH_OBJ,SOLUTION,GRIDS,par)
    J[2,1]=(fh-fl)/(2.0*hh)
    #dF1/dkT
    fl=HH_FOC_T_Def(z_ind,kN,kT,x[1],x[2]-hh,HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_T_Def(z_ind,kN,kT,x[1],x[2]+hh,HH_OBJ,SOLUTION,GRIDS,par)
    J[2,2]=(fh-fl)/(2.0*hh)
    return nothing
end

function F_and_Jacobian_Rep!(x::Array{Float64,1},F::Array{Float64,1},J::Array{Float64,2},
                             z_ind::Int64,kN::Float64,kT::Float64,b::Float64,bprime::Float64,
                             HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    hh=1e-6
    #Update F
    F[1]=HH_FOC_N_Rep(z_ind,kN,kT,b,x[1],x[2],bprime,HH_OBJ,SOLUTION,GRIDS,par)
    F[2]=HH_FOC_T_Rep(z_ind,kN,kT,b,x[1],x[2],bprime,HH_OBJ,SOLUTION,GRIDS,par)

    #Update J
    #dFN/dkN
    fl=HH_FOC_N_Rep(z_ind,kN,kT,b,x[1]-hh,x[2],bprime,HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_N_Rep(z_ind,kN,kT,b,x[1]+hh,x[2],bprime,HH_OBJ,SOLUTION,GRIDS,par)
    J[1,1]=(fh-fl)/(2.0*hh)
    #dFN/dkT
    fl=HH_FOC_N_Rep(z_ind,kN,kT,b,x[1],x[2]-hh,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_N_Rep(z_ind,kN,kT,b,x[1],x[2]+hh,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    J[1,2]=(fh-fl)/(2.0*hh)

    #dFT/dkN
    fl=HH_FOC_T_Rep(z_ind,kN,kT,b,x[1]-hh,x[2],bprime,HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_T_Rep(z_ind,kN,kT,b,x[1]+hh,x[2],bprime,HH_OBJ,SOLUTION,GRIDS,par)
    J[2,1]=(fh-fl)/(2.0*hh)
    #dFT/dkT
    fl=HH_FOC_T_Rep(z_ind,kN,kT,b,x[1],x[2]-hh,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    fh=HH_FOC_T_Rep(z_ind,kN,kT,b,x[1],x[2]+hh,bprime,HH_OBJ,SOLUTION,GRIDS,par)
    J[2,2]=(fh-fl)/(2.0*hh)
    return nothing
end

function HHOptim_Def!(I::CartesianIndex,HH_OBJ::HH_itpObjects,
                     SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack kNlowOpt, kTlowOpt = par
    @unpack GR_kN, GR_kT, GR_z = GRIDS
    (kT_ind,kN_ind,z_ind)=Tuple(I)
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; z=GR_z[z_ind]

    #Start with previous policy as guess
    x0=[SOLUTION.kNprime_D[I], SOLUTION.kTprime_D[I]]
    # x0=[kN, kT]

    #Use Newton's algorithm
    α0=0.1 #slow jumps in direction suggested by Newton
    F=Array{Float64,1}(undef,2)
    J=Array{Float64,2}(undef,2,2)
    F_and_Jacobian_Def!(x0,F,J,z_ind,kN,kT,HH_OBJ,SOLUTION,GRIDS,par)
    x1=x0-J\F

    #Check that did not overshoot to very low capital
    #If it did overshoot then choose midpoint until x1>0
    while x1[1]<kNlowOpt || x1[2]<kTlowOpt
        x1=α0*x1+(1.0-α0)*x0
    end

    #Check that did not overshoot to negative consumption
    #If it did overshoot then choose midpoint until c>0
    while ConsNet(FinalOutput(zDefault(z,par),kN,kT,0.0,par),kN,kT,x1[1],x1[2],par)<=0.0
        x1=α0*x1+(1.0-α0)*x0
    end

    dst=maximum(abs.(x1-x0))
    rdst=100*maximum(abs.((x1-x0)./x0))
    cnt=1

    while dst>1e-2 && cnt<50 && rdst>0.5
        x0.=x1
        F_and_Jacobian_Def!(x0,F,J,z_ind,kN,kT,HH_OBJ,SOLUTION,GRIDS,par)
        x1=x0-J\F

        #Check that did not overshoot to very low capital
        #If it did overshoot then choose midpoint until x1>0
        while x1[1]<kNlowOpt || x1[2]<kTlowOpt
            x1=α0*x1+(1.0-α0)*x0
        end

        #Check that did not overshoot to negative consumption
        #If it did overshoot then choose midpoint until c>0
        while ConsNet(FinalOutput(zDefault(z,par),kN,kT,0.0,par),kN,kT,x1[1],x1[2],par)<=0.0
            x1=α0*x1+(1.0-α0)*x0
        end

        dst=maximum(abs.(x1-x0))
        rdst=100*maximum(abs.((x1-x0)./x0))
        cnt=cnt+1
    end
    SOLUTION.kNprime_D[I]=x1[1]
    SOLUTION.kTprime_D[I]=x1[2]
    return nothing
end

function HHOptim_Rep(bpol::Float64,I::CartesianIndex,HH_OBJ::HH_itpObjects,
                     SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack kNlowOpt, kTlowOpt, γ, κ = par
    @unpack GR_kN, GR_kT, GR_b, GR_z = GRIDS
    @unpack itp_Tr, itp_q1 = SOLUTION
    (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; b=GR_b[b_ind]; z=GR_z[z_ind]

    #Start with previous policy as guess
    x0=[SOLUTION.kNprime[I], SOLUTION.kTprime[I]]

    #Start with low kN' and kT' to approach from the left
    # x0=0.95*[SOLUTION.kNprime[I], SOLUTION.kTprime[I]]
    # x0=[kN, kT]

    #Use Newton's algorithm
    α0=0.1 #slow jumps in direction suggested by Newton
    F=Array{Float64,1}(undef,2)
    J=Array{Float64,2}(undef,2,2)
    F_and_Jacobian_Rep!(x0,F,J,z_ind,kN,kT,b,bpol,HH_OBJ,SOLUTION,GRIDS,par)
    x1=x0-J\F

    #Check that did not overshoot to negative capital
    #If it did overshoot then choose midpoint until x1>0
    while minimum(x1)<=0.0
        x1=α0*x1+(1.0-α0)*x0
    end

    dst=maximum(abs.(x1-x0))
    rdst=100*maximum(abs.((x1-x0)./x0))
    cnt=1

    while dst>1e-2 && cnt<50 && rdst>0.5
        x0.=x1
        F_and_Jacobian_Rep!(x0,F,J,z_ind,kN,kT,b,bpol,HH_OBJ,SOLUTION,GRIDS,par)
        x1=x0-J\F

        #Check that did not overshoot to negative capital
        #If it did overshoot then choose midpoint until x1>0
        while minimum(x1)<=0.0
            x1=α0*x1+(1.0-α0)*x0
        end

        dst=maximum(abs.(x1-x0))
        rdst=100*maximum(abs.((x1-x0)./x0))
        cnt=cnt+1
    end
    return x1[1], x1[2]
end

#Update equilibrium objects in decentralized economy
function ValueInDefault_DEC!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, cmin = par
    @unpack itp_EV, itp_EVD = SOLUTION
    @unpack GR_kT, GR_kN, GR_z = GRIDS
    #Update kN' kT'. It works because HHOptim_Def does not use
    #SOLUTION.kNprime, SOLUTION.kTprime, it uses HH_OBJ which already computed consumption
    #using these, so their mutation does not affect subsequent iterations
    HHOptim_Def!(I,HH_OBJ,SOLUTION,GRIDS,par)

    (kT_ind,kN_ind,z_ind)=Tuple(I)
    kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; z=GR_z[z_ind]

    #Compute output, consumption and value
    zD=zDefault(z,par)
    y=FinalOutput(zD,kN,kT,0.0,par)
    cons=ConsNet(y,kN,kT,SOLUTION.kNprime_D[I],SOLUTION.kTprime_D[I],par)
    if cons>0.0
        SOLUTION.VD[I]=Utility(cons,par)+β*θ*itp_EV(0.0,min(SOLUTION.kTprime_D[I],GR_kT[end]),min(SOLUTION.kNprime_D[I],GR_kN[end]),z)+β*(1.0-θ)*itp_EVD(min(SOLUTION.kTprime_D[I],GR_kT[end]),min(SOLUTION.kNprime_D[I],GR_kN[end]),z)
    else
        SOLUTION.VD[I]=Utility(cmin,par)-SOLUTION.kNprime_D[I]-SOLUTION.kTprime_D[I]
        return nothing
    end
end

function UpdateDefault_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT = par
    @unpack GR_kN, GR_kT, GR_z = GRIDS
    @unpack kNprime_D, kTprime_D = SOLUTION
    #Loop over all states to update policy functions
    for I in CartesianIndices(SOLUTION.VD)
        ValueInDefault_DEC!(I,HH_OBJ,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_kNprime_D=CreateInterpolation_Policies(SOLUTION.kNprime_D,true,GRIDS)
    SOLUTION.itp_kTprime_D=CreateInterpolation_Policies(SOLUTION.kTprime_D,true,GRIDS)
    #Loop over all states to compute expectations over p and z
    for I in CartesianIndices(SOLUTION.EVD)
        Expectation_Default!(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

function ValueInRepayment_DEC(I::CartesianIndex,bprime::Float64,HH_OBJ::HH_itpObjects,
                                 SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, κ, cmin = par
    @unpack itp_EV, itp_q1 = SOLUTION
    @unpack GR_kN, GR_kT, GR_b, GR_z = GRIDS
    (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
    b=GR_b[b_ind]; kN=GR_kN[kN_ind]; kT=GR_kT[kT_ind]; z=GR_z[z_ind]

    #Get use HH's policies for current b'
    kNpol, kTpol=HHOptim_Rep(bprime,I,HH_OBJ,SOLUTION,GRIDS,par)

    #Compute output
    qq=itp_q1(bprime,kTpol,kNpol,z)
    T=qq*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
    y=FinalOutput(z,kN,kT,T,par)
    if y>0.0
        #Compute consumption
        cons=ConsNet(y,kN,kT,kNpol,kTpol,par)
        if cons>cmin
            vv=Utility(cons,par)+β*itp_EV(max(bprime,0.0),min(kTpol,GR_kT[end]),min(kNpol,GR_kN[end]),z)
            return vv
        else
            return Utility(cmin,par)
        end
    else
        return Utility(cmin,par)
    end
end

function OptimInRepayment_DEC!(I::CartesianIndex,HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack blowOpt, bhighOpt, β, γ, κ, cmin = par
    @unpack GR_kN, GR_kT, GR_b, GR_z = GRIDS
    @unpack itp_q1 = SOLUTION
    #ValueInRepayment_DEC computes (kNprime,kTprime) given b'
    foo(bprime::Float64)=-ValueInRepayment_DEC(I,bprime,HH_OBJ,SOLUTION,GRIDS,par)

    #Find bounds for feasible bprime
    #(i.e. b' that yields y>0 and (kN',kT') such that c>0)
    #Lower bound
    i=1
    while foo(GR_b[i])==Utility(cmin,par)
        i=i+1
    end
    if i==1
        blow=blowOpt
    else
        blow=GR_b[i]
    end

    #Upper bound: don't consider too high b'
    #Once its high enough such that q(b')=0 then stop
    z=GRIDS.GR_z[I[4]]
    if i==length(GR_b)
        bhigh=bhighOpt
    else
        i=i+1
        while i<length(GR_b)
            kNpol, kTpol=HHOptim_Rep(GR_b[i],I,HH_OBJ,SOLUTION,GRIDS,par)
            qq=itp_q1(GR_b[i],kTpol,kNpol,z)
            if qq==0.0
                break
            else
                i=i+1
            end
        end
        if i==length(GR_b)
            bhigh=bhighOpt
        else
            bhigh=GR_b[i]
        end
    end

    res=optimize(foo,blow,bhigh,GoldenSection(),
                 abs_tol = 1e-3,iterations=50)

    #Use solution
    if Optim.minimizer(res)<0.0
        SOLUTION.bprime[I]=0.0
        SOLUTION.VP[I]=-foo(0.0)
    else
        SOLUTION.bprime[I]=Optim.minimizer(res)
        SOLUTION.VP[I]=-Optim.minimum(res)
    end
    SOLUTION.kNprime[I], SOLUTION.kTprime[I]=HHOptim_Rep(SOLUTION.bprime[I],I,HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

function UpdateRepayment_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nz, NkN, NkT, Nb, γ, κ = par
    @unpack GR_kN, GR_kT, GR_b, GR_z = GRIDS
    @unpack bprime, itp_q1 = SOLUTION
    #Loop over all states to fill value of repayment and (kN',kT',b')
    #This must be done after doing all capital choices
    for I in CartesianIndices(SOLUTION.VP)
        (b_ind,kT_ind,kN_ind,z_ind)=Tuple(I)
        #Update VP and bprime
        OptimInRepayment_DEC!(I,HH_OBJ,SOLUTION,GRIDS,par)
        #Update V and Tr
        if SOLUTION.VP[I]<SOLUTION.VD[kT_ind,kN_ind,z_ind]
            SOLUTION.V[I]=SOLUTION.VD[kT_ind,kN_ind,z_ind]
            SOLUTION.Tr[I]=0.0
        else
            SOLUTION.V[I]=SOLUTION.VP[I]
            qq=itp_q1(SOLUTION.bprime[I],SOLUTION.kTprime[I],SOLUTION.kNprime[I],GR_z[z_ind])
            SOLUTION.Tr[I]=qq*(SOLUTION.bprime[I]-(1.0-γ)*GR_b[b_ind])-(γ+κ*(1.0-γ))*GR_b[b_ind]
        end
    end
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kNprime=CreateInterpolation_Policies(SOLUTION.kNprime,false,GRIDS)
    SOLUTION.itp_kTprime=CreateInterpolation_Policies(SOLUTION.kTprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_Policies(SOLUTION.bprime,false,GRIDS)
    SOLUTION.itp_Tr=CreateInterpolation_Policies(SOLUTION.Tr,false,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    for I in CartesianIndices(SOLUTION.EV)
        Expectation_Repayment!(I,SOLUTION,GRIDS,par)
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

function UpdateSolution_DEC!(HH_OBJ::HH_itpObjects,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #HH_Obj is already consistent with the previous iteration
    #For the first, it is initiated to be consistent with the end of time
    #So that kN' and kT' are updated using these
    UpdateDefault_DEC!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateRepayment_DEC!(HH_OBJ,SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    UpdateHH_Obj!(HH_OBJ,SOLUTION,GRIDS,par)
    return nothing
end

#Solve decentralized model
function SolveAndSaveModel_VFI_DEC(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTol, Tolpct_q, cnt_max = par
    println("Preparing solution guess")
    SOLUTION_CURRENT=SolutionEndOfTime(GRIDS,par)
    HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    println("Starting VFI")
    while ((dst_V>Tol_V && rdts_V>relTol) || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution_DEC!(HH_OBJ,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        rdts_V=max(rdst_D,rdst_P)
        dst_V=rdst_D
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        println("cnt=$cnt, rdst_D=$rdst_D%, rdst_P=$rdst_P%, dst_q=$dst_q")
        println("    $cnt,  dst_D=$dst_D ,  dst_P=$dst_P ,       $NotConvPct% of q not converged")
    end
    SaveSolution(SOLUTION_NEXT)
    println("Compute Moments")
    ComputeAndSaveMoments(SOLUTION_NEXT,GRIDS,par)
    return nothing
end

function SolveModel_VFI_DEC(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTol, Tolpct_q, cnt_max = par
    println("Preparing solution guess")
    SOLUTION_CURRENT=SolutionEndOfTime(GRIDS,par)
    HH_OBJ=InitiateHH_Obj(SOLUTION_CURRENT,GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    println("Starting VFI")
    while ((dst_V>Tol_V && rdts_V>relTol) || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution_DEC!(HH_OBJ,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        rdts_V=max(rdst_D,rdst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        println("cnt=$cnt, rdst_D=$rdst_D%, rdst_V=$rdst_P%, dst_q=$dst_q")
        println("    $cnt,  dst_D=$dst_D ,  dst_V=$dst_P ,       $NotConvPct% of q not converged")
    end
    return SOLUTION_NEXT
end

###############################################################################
#Functions to compute optimal tax
###############################################################################
function Optimal_Tax(z::Float64,kN::Float64,kT::Float64,b::Float64,
                    SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ = par
    @unpack itp_kNprime, itp_kTprime, itp_bprime, itp_q1 = SOLUTION
    kNprime=itp_kNprime(b,kT,kN,z)
    kTprime=itp_kTprime(b,kT,kN,z)
    bprime=itp_bprime(b,kT,kN,z)
    #Derivative of q
    hh=1e-6
    dq_dkT=(itp_q1(bprime,kTprime+hh,kNprime,z)-itp_q1(bprime,kTprime-hh,kNprime,z))/(2*hh)
    dq_dkN=(itp_q1(bprime,kTprime,kNprime+hh,z)-itp_q1(bprime,kTprime,kNprime-hh,z))/(2*hh)
    #Price of final good
    qq=itp_q1(bprime,kTprime,kNprime,z)
    T=qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
    Pt=PriceFinalGood(z,kN,kT,T,par)
    return (dq_dkT-dq_dkN)*(bprime-(1-γ)*b)/Pt
end

function Optimal_Subsidies(z::Float64,kN::Float64,kT::Float64,b::Float64,
                           SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ = par
    @unpack itp_kNprime, itp_kTprime, itp_bprime, itp_q1 = SOLUTION
    kNprime=itp_kNprime(b,kT,kN,z)
    kTprime=itp_kTprime(b,kT,kN,z)
    bprime=itp_bprime(b,kT,kN,z)
    #Derivative of q
    hh=1e-6
    dq_dkT=(itp_q1(bprime,kTprime+hh,kNprime,z)-itp_q1(bprime,kTprime-hh,kNprime,z))/(2*hh)
    dq_dkN=(itp_q1(bprime,kTprime,kNprime+hh,z)-itp_q1(bprime,kTprime,kNprime-hh,z))/(2*hh)
    #Price of final good
    qq=itp_q1(bprime,kTprime,kNprime,z)
    T=qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
    Pt=PriceFinalGood(z,kN,kT,T,par)
    return dq_dkN*(bprime-(1-γ)*b)/Pt, dq_dkT*(bprime-(1-γ)*b)/Pt
end

function OptimalTaxSeries(S0::States_TS,SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    T=length(S0.z)
    TAU=Array{Float64,1}(undef,T)
    for t in 1:T
        if S0.Def[t]==1
            TAU[t]=0.0
        else
            z=S0.z[t]
            kN=S0.KN[t]
            kT=S0.KT[t]
            b=S0.B[t]
            TAU[t]=Optimal_Tax(z,kN,kT,b,SOL_PLA,GRIDS,par)
        end
    end
    return TAU
end

function OptimalSubsidiesSeries(S0::States_TS,SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    T=length(S0.z)
    TAU_N=Array{Float64,1}(undef,T)
    TAU_T=Array{Float64,1}(undef,T)
    for t in 1:T
        if S0.Def[t]==1
            TAU_N[t]=0.0
            TAU_T[t]=0.0
        else
            z=S0.z[t]
            kN=S0.KN[t]
            kT=S0.KT[t]
            b=S0.B[t]
            TAU_N[t], TAU_T[t]=Optimal_Subsidies(z,kN,kT,b,SOL_PLA,GRIDS,par)
        end
    end
    return TAU_N, TAU_T
end

@with_kw mutable struct Moments_Tau
    #Initiate them at 0.0 to facilitate average across samples
    #Mean and variance
    AvTau::Float64 = 0.0
    stdTau::Float64 = 0.0
    #Cyclicality
    Corr_GDP_tau::Float64 = 0.0
    Corr_Spreads_tau::Float64 = 0.0
    Corr_RER_tau::Float64 = 0.0
end

function OptimalTaxMomentsOnce(SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    #Get path of states with same criteria as for moments
    S0=GetOnePathForMoments(SOL_PLA,GRIDS,par)
    #Get the corresponding path of taxes
    TAU=OptimalTaxSeries(S0,SOL_PLA,GRIDS,par)

    #Compute other variables
    @unpack itp_kNprime, itp_kTprime, itp_bprime = SOL_PLA
    @unpack itp_kNprime_D, itp_kTprime_D, itp_q1 = SOL_PLA
    @unpack γ, κ, δ, HPFilter_Par = par
    Tmom=length(TAU)
    P_TS=Array{Float64,1}(undef,Tmom)
    Y_TS=Array{Float64,1}(undef,Tmom)
    GDP_TS=Array{Float64,1}(undef,Tmom)
    con_TS=Array{Float64,1}(undef,Tmom)
    inv_TS=Array{Float64,1}(undef,Tmom)
    TB_TS=Array{Float64,1}(undef,Tmom)
    CA_TS=Array{Float64,1}(undef,Tmom)
    while true
        for t in 1:Tmom
            z=S0.z[t]
            kN=S0.KN[t]
            kT=S0.KT[t]
            b=S0.B[t]
            if t<Tmom
                kNprime=S0.KN[t+1]
                kTprime=S0.KT[t+1]
                bprime=S0.B[t+1]
            else
                if S0.Def[t]==0.0
                    kNprime=itp_kNprime(b,kT,kN,z)
                    kTprime=itp_kTprime(b,kT,kN,z)
                    bprime=itp_bprime(b,kT,kN,z)
                else
                    kNprime=itp_kNprime_D(kT,kN,z)
                    kTprime=itp_kTprime_D(kT,kN,z)
                    bprime=0.0
                end
            end
            if S0.Def[t]==0.0
                q=itp_q1(bprime,kTprime,kNprime,z)
                X=q*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
                y=FinalOutput(z,kN,kT,X,par)
                P=PriceFinalGood(z,kN,kT,X,par)
            else
                zD=zDefault(z,par)
                X=0.0
                y=FinalOutput(zD,kN,kT,X,par)
                P=PriceFinalGood(zD,kN,kT,X,par)
            end
            P_TS[t]=P
            Y_TS[t]=y
            GDP_TS[t]=P*y-X
            AdjCost=CapitalAdjustment(kNprime,kN,par)+CapitalAdjustment(kTprime,kT,par)
            inv_TS[t]=P*((kNprime+kTprime)-(1.0-δ)*(kN+kT))
            con_TS[t]=GDP_TS[t]-inv_TS[t]+X-P*AdjCost
            TB_TS[t]=-X
            CA_TS[t]=-(bprime-b)
        end
        #reject samples with negative consumption
        if minimum(abs.(con_TS))>=0.0
            break
        else
            S0=GetOnePathForMoments(SOL_PLA,GRIDS,par)
            TAU=OptimalTaxSeries(S0,SOL_PLA,GRIDS,par)
        end
    end

    #Mean and variance
    AvTau=mean(100*TAU)
    stdTau=std(100*TAU)
    #Cyclicality
    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(GDP_TS))
    GDP_trend=hp_filter(log_GDP,HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    Corr_GDP_tau=cor(100*TAU,GDP_cyc)
    Corr_Spreads_tau=cor(100*TAU,S0.Spreads)
    Corr_RER_tau=cor(100*TAU,1 ./P_TS)
    return Moments_Tau(AvTau,stdTau,Corr_GDP_tau,Corr_Spreads_tau,Corr_RER_tau)
end

function AverageTauMomentsManySamples(SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    Random.seed!(1234)
    #Initiate them at 0.0 to facilitate average across samples
    MOMENTS=Moments_Tau()
    for i in 1:NSamplesMoments
        # println("Sample $i for moments")
        MOMS=OptimalTaxMomentsOnce(SOL_PLA,GRIDS,par)
        #Default, spreads, and Debt
        MOMENTS.AvTau=MOMENTS.AvTau+MOMS.AvTau/NSamplesMoments
        MOMENTS.stdTau=MOMENTS.stdTau+MOMS.stdTau/NSamplesMoments
        MOMENTS.Corr_GDP_tau=MOMENTS.Corr_GDP_tau+MOMS.Corr_GDP_tau/NSamplesMoments
        MOMENTS.Corr_Spreads_tau=MOMENTS.Corr_Spreads_tau+MOMS.Corr_Spreads_tau/NSamplesMoments
        MOMENTS.Corr_RER_tau=MOMENTS.Corr_RER_tau+MOMS.Corr_RER_tau/NSamplesMoments
    end
    return MOMENTS
end

################################################################################
### Functions to match moments with planner
################################################################################
function Setup_MomentMatching(φ::Float64,d0::Float64,d1::Float64,β::Float64)
    #Setup parameters
    par=Pars(φ=φ,d0=d0,d1=d1,β=β)
    #Setup Grids
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

function SolveModel_VFI_ForMomentMatching(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTol, Tolpct_q, cnt_max = par
    SOLUTION_CURRENT=SolutionEndOfTime(GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdts_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while ((dst_V>Tol_V && rdts_V>relTol) || (dst_q>Tol_q && NotConvPct>Tolpct_q)) && cnt<cnt_max
        UpdateSolution!(SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistanceV(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_V=max(dst_D,dst_P)
        rdts_V=max(rdst_D,rdst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
    end
    #Return moments
    return AverageMomentsManySamples(SOLUTION_NEXT,GRIDS,par)
end

function CheckMomentsForTry(PARS_TRY::Array{Float64,1})
    φ=PARS_TRY[1]
    d1=PARS_TRY[2]
    knk=PARS_TRY[3]
    β=PARS_TRY[4]
    d0=-knk*d1
    # par, GRIDS=Setup_MomentMatching(φ,d0,d1,β)
    par, GRIDS=Setup(β,φ,d0,d1)
    MOM_VEC=Array{Float64,1}(undef,16)
    MOMENTS=SolveModel_VFI_ForMomentMatching(GRIDS,par)
    MOM_VEC[1]=MOMENTS.DefaultPr
    MOM_VEC[2]=MOMENTS.MeanSpreads
    MOM_VEC[3]=MOMENTS.StdSpreads
    MOM_VEC[4]=MOMENTS.Debt_GDP
    MOM_VEC[5]=MOMENTS.kN_GDP
    MOM_VEC[6]=MOMENTS.kT_GDP
    MOM_VEC[7]=MOMENTS.AvλK
    MOM_VEC[8]=MOMENTS.stdλK
    MOM_VEC[9]=MOMENTS.σ_GDP
    MOM_VEC[10]=MOMENTS.σ_con
    MOM_VEC[11]=MOMENTS.σ_inv
    MOM_VEC[12]=MOMENTS.Corr_con_GDP
    MOM_VEC[13]=MOMENTS.Corr_inv_GDP
    MOM_VEC[14]=MOMENTS.Corr_Spreads_GDP
    MOM_VEC[15]=MOMENTS.Corr_CA_GDP
    MOM_VEC[16]=MOMENTS.Corr_TB_GDP
    return MOM_VEC
end

function CalibrateMatchingMoments(N::Int64,lb::Vector{Float64},ub::Vector{Float64})
    #Generate Sobol sequence of vectors (α,d1,knk,β)
    ss = skip(SobolSeq(lb, ub),N)
    MAT_TRY=Array{Float64,2}(undef,N,4)
    for i in 1:N
        MAT_TRY[i,:]=next!(ss)
    end
    #Loop paralelly over all parameter tries
    DistVector=SharedArray{Float64,1}(N)
    #There are 16 moments, columns should be 16 + 4 (number of parameters)
    PARAMETER_MOMENTS_MATRIX=SharedArray{Float64,2}(N,20)
    @sync @distributed for i in 1:N
        println("Doing i=$i of $N")
        PARAMETER_MOMENTS_MATRIX[i,1:4]=MAT_TRY[i,:]
        PARAMETER_MOMENTS_MATRIX[i,5:20]=CheckMomentsForTry(MAT_TRY[i,:])
    end
    COL_NAMES=["phi" "d1" "knk" "beta" "DefaultPr" "MeanSpreads" "StdSpreads" "Debt_GDP" "kN_GDP" "kT_GDP" "AvλK" "stdλK" "σ_GDP" "σ_con" "σ_inv" "Corr_con_GDP" "Corr_inv_GDP" "Corr_Spreads_GDP" "Corr_CA_GDP" "Corr_TB_GDP"]
    MAT=[COL_NAMES; PARAMETER_MOMENTS_MATRIX]
    writedlm("TriedCalibrations.csv",MAT,',')
    return nothing
end


# include("PlotsCrisis.jl")
