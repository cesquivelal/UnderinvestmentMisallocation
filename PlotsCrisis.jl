
@with_kw mutable struct Paths
    #States
    z::Array{Float64,1}
    Def::Array{Float64,1}
    KN::Array{Float64,1}
    KT::Array{Float64,1}
    B::Array{Float64,1}
    #Variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    InvN::Array{Float64,1}
    InvT::Array{Float64,1}
    Cons::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}
    RER::Array{Float64,1}
    TAU_N::Array{Float64,1}
    TAU_T::Array{Float64,1}
end

###############################################################################
#Functions to analyze debt crises
###############################################################################
function GetInitialStateAndPathOfShocks(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, TsinceDefault, Tsim = par
    @unpack t_before_crisis, t_after_crisis = par
    #Get initial state and path after TsinceDefault with no default
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
        if t0+t_before_crisis+t_after_crisis+10==length(STATES_TS.Def)
            NLongSamples=NLongSamples+1
            if NLongSamples<=5
                #Get another long sample
                STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
                t0=drp+1
                tsince=0
            else
                println("failed to get T since default in good standing")
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
            #Found t0 of TsinceDefault in good standing
            break
        end
    end
    return STATES_TS.z[t0:t0+t_before_crisis+t_after_crisis], STATES_TS.Def[t0], STATES_TS.KN[t0], STATES_TS.KT[t0], STATES_TS.B[t0]
end

function SimulateStates_GivenShocks(z::Array{Float64},Def_1::Float64,kN0::Float64,kT0::Float64,
                                    b0::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, γ, κ, θ = par
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
    @unpack itp_kNprime_D, itp_kTprime_D = SOLUTION
    #Initiate vectors
    Tsim=length(z)
    Def=Array{Float64,1}(undef,Tsim)
    KN=Array{Float64,1}(undef,Tsim)
    KT=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            KN[t]=kN0
            KT[t]=kT0
            B[t]=b0
            if Def_1==0.0
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
                        Spreads[t]=0.0
                    end
                else
                    Def[t]=1.0
                    KN[t+1]=itp_kNprime_D(KT[t],KN[t],z[t])
                    KT[t+1]=itp_kTprime_D(KT[t],KN[t],z[t])
                    B[t+1]=0.0
                    Spreads[t]=0.0
                end
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

function GetOnePathCrisisStates(CrisisSpread::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack t_before_crisis, t_after_crisis = par

    zTS, Def_1, kN0, kT0, b0 = GetInitialStateAndPathOfShocks(SOLUTION,GRIDS,par)
    STATES_TS=SimulateStates_GivenShocks(zTS,Def_1,kN0,kT0,b0,SOLUTION,GRIDS,par)

    t_crisis=t_before_crisis+1
    tries=0
    while true
        #Get crisis without default
        if STATES_TS.Spreads[t_crisis]>=CrisisSpread && sum(STATES_TS.Def)==0
            break
        else
            #Get a new sample
            tries=tries+1
            zTS, Def_1, kN0, kT0, b0 = GetInitialStateAndPathOfShocks(SOLUTION,GRIDS,par)
            STATES_TS=SimulateStates_GivenShocks(zTS,Def_1,kN0,kT0,b0,SOLUTION,GRIDS,par)
            if tries>1000
                println("tried 500 times, didn't find crisis without default")
                break
            end
        end
    end

    return STATES_TS
end

function GetInitialStateSinceDefault(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, TsinceDefault, Tsim = par
    #Get initial state and path after TsinceDefault with no default
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
        if t0+10==length(STATES_TS.Def)
            NLongSamples=NLongSamples+1
            if NLongSamples<=5
                #Get another long sample
                STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
                t0=drp+1
                tsince=0
            else
                println("failed to get T since default in good standing")
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
            #Found t0 of TsinceDefault in good standing
            break
        end
    end
    return STATES_TS.z[t0], STATES_TS.Def[t0], STATES_TS.KN[t0], STATES_TS.KT[t0], STATES_TS.B[t0]
end

function GetCrisisPathFromStates(S_CRISIS::States_TS,SOLUTION::Solution,SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    #Get path of states
    TAU_N, TAU_T=OptimalSubsidiesSeries(S_CRISIS,SOL_PLA,GRIDS,par)

    #Compute variables
    T=length(S_CRISIS.z)
    #States
    z=S_CRISIS.z
    Def=S_CRISIS.Def
    KN=S_CRISIS.KN
    KT=S_CRISIS.KT
    B=S_CRISIS.B
    #Variables
    Spreads=S_CRISIS.Spreads
    GDP=Array{Float64,1}(undef,T)
    InvN=Array{Float64,1}(undef,T)
    InvT=Array{Float64,1}(undef,T)
    Cons=Array{Float64,1}(undef,T)
    TB=Array{Float64,1}(undef,T)
    CA=Array{Float64,1}(undef,T)
    RER=Array{Float64,1}(undef,T)

    @unpack itp_q1 = SOLUTION
    @unpack γ, κ, δ = par

    for t in 1:T
        if t==T
            @unpack itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
            kNprime=itp_kNprime(B[t],KT[t],KN[t],z[t])
            kTprime=itp_kTprime(B[t],KT[t],KN[t],z[t])
            bprime=itp_bprime(B[t],KT[t],KN[t],z[t])
        else
            kNprime=KN[t+1]
            kTprime=KT[t+1]
            bprime=B[t+1]
        end
        qq=itp_q1(bprime,kTprime,kNprime,z[t])
        Tr=qq*(bprime-(1-γ)*B[t])-(γ+(1-γ)*κ)*B[t]
        GDP[t]=FinalOutput(z[t],KN[t],KT[t],Tr,par)
        RER[t]=1.0/PriceFinalGood(z[t],KN[t],KT[t],Tr,par)
        InvN[t]=kNprime-(1-δ)*KN[t]
        InvT[t]=kTprime-(1-δ)*KT[t]
        Cons[t]=ConsNet(GDP[t],KN[t],KT[t],kNprime,kTprime,par)
        TB[t]=-Tr
        CA[t]=-(bprime-B[t])
    end
    return Paths(z,Def,KN,KT,B,Spreads,GDP,InvN,InvT,Cons,TB,CA,RER,TAU_N,TAU_T)
end

function GetBothPathsCrisisStates(CrisisSpread::Float64,SOL_PLA::Solution,SOL_DEC::Solution,GRIDS::Grids,par::Pars)
    @unpack t_before_crisis, t_after_crisis = par

    #Get one crisis path in decentralized
    SD=GetOnePathCrisisStates(CrisisSpread,SOL_DEC,GRIDS,par)
    CrD=GetCrisisPathFromStates(SD,SOL_DEC,SOL_PLA,GRIDS,par)

    #Get planner's path
    #Add Nadd periods with zD[1]
    Nadd=10
    zP=vcat(ones(Float64,Nadd)*SD.z[1],SD.z)
    zP[1], DefP0, kN0, kT0, b0 =GetInitialStateSinceDefault(SOL_PLA,GRIDS,par)
    #Simulate using same shocks from decentralized crisis but
    #initial state from planner's ergodic distribution
    SP_=SimulateStates_GivenShocks(zP,DefP0,kN0,kT0,b0,SOL_PLA,GRIDS,par)
    SP=States_TS(SP_.z[Nadd+1:end],SP_.Def[Nadd+1:end],SP_.KN[Nadd+1:end],SP_.KT[Nadd+1:end],SP_.B[Nadd+1:end],SP_.Spreads[Nadd+1:end])

    #Make sure SP does not default either
    cnt=1
    while sum(SP.Def)>0
        #Get one crisis path in decentralized
        SD=GetOnePathCrisisStates(CrisisSpread,SOL_DEC,GRIDS,par)
        CrD=GetCrisisPathFromStates(SD,SOL_DEC,SOL_PLA,GRIDS,par)

        #Get planner's path
        #Add Nadd periods with zD[1]
        Nadd=10
        zP=vcat(ones(Float64,Nadd)*SD.z[1],SD.z)
        zP[1], DefP0, kN0, kT0, b0 =GetInitialStateSinceDefault(SOL_PLA,GRIDS,par)
        #Simulate using same shocks from decentralized crisis but
        #initial state from planner's ergodic distribution
        SP_=SimulateStates_GivenShocks(zP,DefP0,kN0,kT0,b0,SOL_PLA,GRIDS,par)
        SP=States_TS(SP_.z[Nadd+1:end],SP_.Def[Nadd+1:end],SP_.KN[Nadd+1:end],SP_.KT[Nadd+1:end],SP_.B[Nadd+1:end],SP_.Spreads[Nadd+1:end])
        cnt=cnt+1
        if cnt>10
            println("Could not find path where neither P or D default")
            break
        end
    end

    CrP=GetCrisisPathFromStates(SP,SOL_PLA,SOL_PLA,GRIDS,par)

    return CrD, CrP
end

function GetOneCrisisPath(CrisisSpread::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Get path of states
    S_CRISIS=GetOnePathCrisisStates(CrisisSpread,SOLUTION,GRIDS,par)
    TAU_N, TAU_T=OptimalSubsidiesSeries(S_CRISIS,SOLUTION,GRIDS,par)

    #Compute variables
    T=length(S_CRISIS.z)
    #States
    z=S_CRISIS.z
    Def=S_CRISIS.Def
    KN=S_CRISIS.KN
    KT=S_CRISIS.KT
    B=S_CRISIS.B
    #Variables
    Spreads=S_CRISIS.Spreads
    GDP=Array{Float64,1}(undef,T)
    InvN=Array{Float64,1}(undef,T)
    InvT=Array{Float64,1}(undef,T)
    Cons=Array{Float64,1}(undef,T)
    TB=Array{Float64,1}(undef,T)
    CA=Array{Float64,1}(undef,T)
    RER=Array{Float64,1}(undef,T)

    @unpack itp_q1 = SOLUTION
    @unpack γ, κ, δ = par

    for t in 1:T
        if t==T
            @unpack itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
            kNprime=itp_kNprime(B[t],KT[t],KN[t],z[t])
            kTprime=itp_kTprime(B[t],KT[t],KN[t],z[t])
            bprime=itp_bprime(B[t],KT[t],KN[t],z[t])
        else
            kNprime=KN[t+1]
            kTprime=KT[t+1]
            bprime=B[t+1]
        end
        qq=itp_q1(bprime,kTprime,kNprime,z[t])
        Tr=qq*(bprime-(1-γ)*B[t])-(γ+(1-γ)*κ)*B[t]
        GDP[t]=FinalOutput(z[t],KN[t],KT[t],Tr,par)
        RER[t]=1.0/PriceFinalGood(z[t],KN[t],KT[t],Tr,par)
        InvN[t]=kNprime-(1-δ)*KN[t]
        InvT[t]=kTprime-(1-δ)*KT[t]
        Cons[t]=ConsNet(GDP[t],KN[t],KT[t],kNprime,kTprime,par)
        TB[t]=-Tr
        CA[t]=-(bprime-B[t])
    end
    return Paths(z,Def,KN,KT,B,Spreads,GDP,InvN,InvT,Cons,TB,CA,RER,TAU_N,TAU_T)
end

function AverageCrisisPaths(CrisisSpread::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack N_crises = par
    Random.seed!(1234)
    #Do first
    CRISIS=GetOneCrisisPath(CrisisSpread,SOLUTION,GRIDS,par)

    #Add paths
    for i=2:N_crises
        println("Doing crisis path i=$i")
        C1=GetOneCrisisPath(CrisisSpread,SOLUTION,GRIDS,par)
        CRISIS.z=CRISIS.z .+ C1.z
        CRISIS.Def=CRISIS.Def .+ C1.Def
        CRISIS.KN=CRISIS.KN .+ C1.KN
        CRISIS.KT=CRISIS.KT .+ C1.KT
        CRISIS.B=CRISIS.B .+ C1.B
        #Variables
        CRISIS.Spreads=CRISIS.Spreads .+ C1.Spreads
        CRISIS.GDP=CRISIS.GDP .+ C1.GDP
        CRISIS.InvN=CRISIS.InvN .+ C1.InvN
        CRISIS.InvT=CRISIS.InvT .+ C1.InvT
        CRISIS.Cons=CRISIS.Cons .+ C1.Cons
        CRISIS.TB=CRISIS.TB .+ C1.TB
        CRISIS.CA=CRISIS.CA .+ C1.CA
        CRISIS.RER=CRISIS.RER .+ C1.RER
        CRISIS.TAU_N=CRISIS.TAU_N .+ C1.TAU_N
        CRISIS.TAU_T=CRISIS.TAU_T .+ C1.TAU_T
    end
    #Average
    CRISIS.z=CRISIS.z ./ N_crises
    CRISIS.Def=CRISIS.Def ./ N_crises
    CRISIS.KN=CRISIS.KN ./ N_crises
    CRISIS.KT=CRISIS.KT ./ N_crises
    CRISIS.B=CRISIS.B ./ N_crises
    #Variables
    CRISIS.Spreads=CRISIS.Spreads ./ N_crises
    CRISIS.GDP=CRISIS.GDP ./ N_crises
    CRISIS.InvN=CRISIS.InvN ./ N_crises
    CRISIS.InvT=CRISIS.InvT ./ N_crises
    CRISIS.Cons=CRISIS.Cons ./ N_crises
    CRISIS.TB=CRISIS.TB ./ N_crises
    CRISIS.CA=CRISIS.CA ./ N_crises
    CRISIS.RER=CRISIS.RER ./ N_crises
    CRISIS.TAU_N=CRISIS.TAU_N ./ N_crises
    CRISIS.TAU_T=CRISIS.TAU_T ./ N_crises
    return CRISIS
end

function AverageBothCrisiesPaths(CrisisSpread::Float64,SOL_PLA::Solution,SOL_DEC::Solution,GRIDS::Grids,par::Pars)
    @unpack N_crises = par
    Random.seed!(1234)
    #Do first
    CRISIS_D, CRISIS_P=GetBothPathsCrisisStates(CrisisSpread,SOL_PLA,SOL_DEC,GRIDS,par)

    #Add paths
    for i=2:N_crises
        println("Doing crisis path i=$i")
        CD1, CP1=GetBothPathsCrisisStates(CrisisSpread,SOL_PLA,SOL_DEC,GRIDS,par)

        #Sum decentralized
        CRISIS_D.z=CRISIS_D.z .+ CD1.z
        CRISIS_D.Def=CRISIS_D.Def .+ CD1.Def
        CRISIS_D.KN=CRISIS_D.KN .+ CD1.KN
        CRISIS_D.KT=CRISIS_D.KT .+ CD1.KT
        CRISIS_D.B=CRISIS_D.B .+ CD1.B
        CRISIS_D.Spreads=CRISIS_D.Spreads .+ CD1.Spreads
        CRISIS_D.GDP=CRISIS_D.GDP .+ CD1.GDP
        CRISIS_D.InvN=CRISIS_D.InvN .+ CD1.InvN
        CRISIS_D.InvT=CRISIS_D.InvT .+ CD1.InvT
        CRISIS_D.Cons=CRISIS_D.Cons .+ CD1.Cons
        CRISIS_D.TB=CRISIS_D.TB .+ CD1.TB
        CRISIS_D.CA=CRISIS_D.CA .+ CD1.CA
        CRISIS_D.RER=CRISIS_D.RER .+ CD1.RER
        CRISIS_D.TAU_N=CRISIS_D.TAU_N .+ CD1.TAU_N
        CRISIS_D.TAU_T=CRISIS_D.TAU_T .+ CD1.TAU_T

        #Sum planner
        CRISIS_P.z=CRISIS_P.z .+ CP1.z
        CRISIS_P.Def=CRISIS_P.Def .+ CP1.Def
        CRISIS_P.KN=CRISIS_P.KN .+ CP1.KN
        CRISIS_P.KT=CRISIS_P.KT .+ CP1.KT
        CRISIS_P.B=CRISIS_P.B .+ CP1.B
        CRISIS_P.Spreads=CRISIS_P.Spreads .+ CP1.Spreads
        CRISIS_P.GDP=CRISIS_P.GDP .+ CP1.GDP
        CRISIS_P.InvN=CRISIS_P.InvN .+ CP1.InvN
        CRISIS_P.InvT=CRISIS_P.InvT .+ CP1.InvT
        CRISIS_P.Cons=CRISIS_P.Cons .+ CP1.Cons
        CRISIS_P.TB=CRISIS_P.TB .+ CP1.TB
        CRISIS_P.CA=CRISIS_P.CA .+ CP1.CA
        CRISIS_P.RER=CRISIS_P.RER .+ CP1.RER
        CRISIS_P.TAU_N=CRISIS_P.TAU_N .+ CP1.TAU_N
        CRISIS_P.TAU_T=CRISIS_P.TAU_T .+ CP1.TAU_T
    end
    #Average decentralized
    CRISIS_D.z=CRISIS_D.z ./ N_crises
    CRISIS_D.Def=CRISIS_D.Def ./ N_crises
    CRISIS_D.KN=CRISIS_D.KN ./ N_crises
    CRISIS_D.KT=CRISIS_D.KT ./ N_crises
    CRISIS_D.B=CRISIS_D.B ./ N_crises
    CRISIS_D.Spreads=CRISIS_D.Spreads ./ N_crises
    CRISIS_D.GDP=CRISIS_D.GDP ./ N_crises
    CRISIS_D.InvN=CRISIS_D.InvN ./ N_crises
    CRISIS_D.InvT=CRISIS_D.InvT ./ N_crises
    CRISIS_D.Cons=CRISIS_D.Cons ./ N_crises
    CRISIS_D.TB=CRISIS_D.TB ./ N_crises
    CRISIS_D.CA=CRISIS_D.CA ./ N_crises
    CRISIS_D.RER=CRISIS_D.RER ./ N_crises
    CRISIS_D.TAU_N=CRISIS_D.TAU_N ./ N_crises
    CRISIS_D.TAU_T=CRISIS_D.TAU_T ./ N_crises

    #Average planner
    CRISIS_P.z=CRISIS_P.z ./ N_crises
    CRISIS_P.Def=CRISIS_P.Def ./ N_crises
    CRISIS_P.KN=CRISIS_P.KN ./ N_crises
    CRISIS_P.KT=CRISIS_P.KT ./ N_crises
    CRISIS_P.B=CRISIS_P.B ./ N_crises
    CRISIS_P.Spreads=CRISIS_P.Spreads ./ N_crises
    CRISIS_P.GDP=CRISIS_P.GDP ./ N_crises
    CRISIS_P.InvN=CRISIS_P.InvN ./ N_crises
    CRISIS_P.InvT=CRISIS_P.InvT ./ N_crises
    CRISIS_P.Cons=CRISIS_P.Cons ./ N_crises
    CRISIS_P.TB=CRISIS_P.TB ./ N_crises
    CRISIS_P.CA=CRISIS_P.CA ./ N_crises
    CRISIS_P.RER=CRISIS_P.RER ./ N_crises
    CRISIS_P.TAU_N=CRISIS_P.TAU_N ./ N_crises
    CRISIS_P.TAU_T=CRISIS_P.TAU_T ./ N_crises
    return CRISIS_D, CRISIS_P
end

function CreateCrisisPlots(CRISIS::Paths,par::Pars)
    @unpack t_before_crisis, t_after_crisis = par
    @unpack size_width, size_height = par
    #General objects for plots
    xx=[-t_before_crisis:1:t_after_crisis]

    #Plot GDP
    TITLE="GDP"
    YLABEL="percentage change from t=-$t_before_crisis"
    yy=100*((CRISIS.GDP./CRISIS.GDP[1]).-1)
    plt_gdp=plot(xx,yy,xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 legend=false,size=(size_width,size_height))

    #Plot Spreads
    TITLE="spreads"
    YLABEL="percentage points"
    yy=CRISIS.Spreads
    plt_spreads=plot(xx,yy,xlabel="t",
                     title=TITLE,ylabel=YLABEL,
                     legend=false,size=(size_width,size_height))

    #Plot debt
    TITLE="debt"
    YLABEL="percentage of AvGDP"
    yy=100*CRISIS.B./CRISIS.GDP[1]
    plt_b=plot(xx,yy,xlabel="t",
               title=TITLE,ylabel=YLABEL,
               legend=false,size=(size_width,size_height))

    #Plot RER
    TITLE="real exchange rate"
    YLABEL="percentage change from t=-$t_before_crisis"
    yy=100*((CRISIS.RER./CRISIS.RER[1]).-1)
    plt_rer=plot(xx,yy,xlabel="t",
               title=TITLE,ylabel=YLABEL,
               legend=false,size=(size_width,size_height))

    #Plot investment
    TITLE="investment"
    YLABEL="percentage of AvGDP"
    yy=100*(CRISIS.InvN .+ CRISIS.InvT) ./ CRISIS.GDP[1]
    plt_inv=plot(xx,yy,xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 legend=false,size=(size_width,size_height))

    #Plot capital
    # TITLE="fraction of capital in T"
    # YLABEL="kT/(kN+kT)"
    # yy=CRISIS.KT ./ (CRISIS.KT .+ CRISIS.KN)
    # plt_kk=plot(xx,yy,xlabel="t",
    #             title=TITLE,ylabel=YLABEL,
    #             legend=false,size=(size_width,size_height))

    #Plot TAU
    TITLE="optimal subsidies'"
    YLABEL="τkN, τkT"
    yyN=100*CRISIS.TAU_N
    yyT=100*CRISIS.TAU_T
    LABELS_T=["τkN" "τkT"]
    LINECOLOR_T=[:green :orange]
    plt_tau=plot(xx,[yyN,yyT],xlabel="t",label=LABELS_T,
                title=TITLE,ylabel=YLABEL,
                linecolor=LINECOLOR_T,
                legend=:best,size=(size_width,size_height))

    #Create plot array
    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_gdp,plt_spreads,plt_b,plt_rer,plt_inv,plt_tau,
             layout=l,size=(size_width*2,size_height*3))
    # savefig(plt,"$FOLDER_GRAPHS\\RegionsCoupons.png")
    return plt
end

function CreateTwoCrisisPlots(CRISIS_P::Paths,CRISIS_D::Paths,par::Pars)
    @unpack t_before_crisis, t_after_crisis = par
    @unpack size_width, size_height = par
    #General objects for plots
    xx=[-t_before_crisis:1:t_after_crisis]
    LABELS=["Planner" "Decentralized" "0"]
    LINESTYLE=[:solid :dash :solid]
    LINECOLOR=[:blue :red :black]


    #Plot GDP
    TITLE="GDP"
    YLABEL="percentage change"
    # yyP=100*((CRISIS_P.GDP./CRISIS_P.GDP[1]).-1)
    # yyD=100*((CRISIS_D.GDP./CRISIS_D.GDP[1]).-1)
    pP=1 ./ CRISIS_P.RER
    pD=1 ./ CRISIS_D.RER
    gdpP=pP .* CRISIS_P.GDP .+ CRISIS_P.TB
    gdpD=pD .* CRISIS_D.GDP .+ CRISIS_D.TB
    yyP=100*((gdpP./gdpP[1]).-1)
    yyD=100*((gdpD./gdpD[1]).-1)
    zz=zeros(Float64,length(yyP))
    plt_gdp=plot(xx,[yyP yyD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 legend=false,size=(size_width,size_height))

    #Plot GDP
    # TITLE="GDP"
    # YLABEL="level"
    # yyP=CRISIS_P.GDP
    # yyD=CRISIS_D.GDP
    # plt_gdp=plot(xx,[yyP yyD],xlabel="t",
    #              title=TITLE,ylabel=YLABEL,
    #              label=LABELS,
    #              linestyle=LINESTYLE,
    #              linecolor=LINECOLOR,
    #              legend=:best,size=(size_width,size_height))

    #Plot Spreads
    TITLE="change in spreads"
    YLABEL="percentage points"
    yyP=CRISIS_P.Spreads .- CRISIS_P.Spreads[1]
    yyD=CRISIS_D.Spreads .- CRISIS_D.Spreads[1]
    plt_spreads=plot(xx,[yyP yyD zz],xlabel="t",
                     title=TITLE,ylabel=YLABEL,
                     label=LABELS,
                     linestyle=LINESTYLE,
                     linecolor=LINECOLOR,
                     legend=:topright,size=(size_width,size_height))

    #Plot current account
    # TITLE="current account"
    # YLABEL="percentage of GDP"
    # yyP=100*CRISIS_P.CA./CRISIS_P.GDP
    # yyD=100*CRISIS_D.CA./CRISIS_D.GDP
    # plt_b=plot(xx,[yyP yyD],xlabel="t",
    #            title=TITLE,ylabel=YLABEL,
    #            linestyle=LINESTYLE,
    #            linecolor=LINECOLOR,
    #            legend=false,size=(size_width,size_height))

    #Plot debt
    TITLE="change in debt"
    YLABEL="percentage of GDP"
    yyP=100*CRISIS_P.B./CRISIS_P.GDP
    yyD=100*CRISIS_D.B./CRISIS_D.GDP
    plt_b=plot(xx,[yyP.-yyP[1] yyD.-yyD[1] zz],xlabel="t",
               title=TITLE,ylabel=YLABEL,
               linestyle=LINESTYLE,
               linecolor=LINECOLOR,
               legend=false,size=(size_width,size_height))

    #Plot RER
    TITLE="real exchange rate"
    YLABEL="percentage change"
    yyP=100*((CRISIS_P.RER./CRISIS_P.RER[1]).-1)
    yyD=100*((CRISIS_D.RER./CRISIS_D.RER[1]).-1)
    plt_rer=plot(xx,[yyP yyD zz],xlabel="t",
               title=TITLE,ylabel=YLABEL,
               linestyle=LINESTYLE,
               linecolor=LINECOLOR,
               legend=false,size=(size_width,size_height))

    #Plot TB
    TITLE="trade balance"
    YLABEL="percentage of GDP"
    yyP=100*CRISIS_P.TB./gdpP
    yyD=100*CRISIS_D.TB./gdpD
    plt_tb=plot(xx,[yyP.-yyP[1] yyD.-yyD[1] zz],xlabel="t",
               title=TITLE,ylabel=YLABEL,
               linestyle=LINESTYLE,
               linecolor=LINECOLOR,
               legend=false,size=(size_width,size_height))

    #Plot investment
    TITLE="total investment"
    YLABEL="percentage change"
    yyP=CRISIS_P.InvN .+ CRISIS_P.InvT
    yyD=CRISIS_D.InvN .+ CRISIS_D.InvT
    plt_inv=plot(xx,100*[(yyP./yyP[1]).-1 (yyD./yyD[1]).-1 zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 legend=false,size=(size_width,size_height))

    #Plot consumption
    TITLE="consumption"
    YLABEL="percentage change"
    yyP=100*((CRISIS_P.Cons ./ CRISIS_P.Cons[1]) .- 1)
    yyD=100*((CRISIS_D.Cons./CRISIS_D.Cons[1]) .- 1)
    plt_con=plot(xx,[yyP yyD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 legend=false,size=(size_width,size_height))

    #Create plot array
    #Create plot array
    l = @layout([a b c; d e f])
    plt=plot(plt_spreads,plt_tb,plt_rer,plt_inv,plt_gdp,plt_con,
             layout=l,size=(size_width*3,size_height*2))
    FOLDER_GRAPHS="Plots"
    savefig(plt,"$FOLDER_GRAPHS\\CrisisPlots.pdf")
    return plt
end

function CreateTwoCrisisPlotsSectors(CRISIS_P::Paths,CRISIS_D::Paths,par::Pars)
    @unpack t_before_crisis, t_after_crisis = par
    @unpack size_width, size_height = par
    #General objects for plots
    xx=[-t_before_crisis:1:t_after_crisis]
    LINESTYLE=[:solid :dash :solid]
    LINECOLOR=[:blue :red :black]

    #Plot investment in N and T
    yyNP=100*((CRISIS_P.InvN ./ CRISIS_P.InvN[1]) .- 1)
    yyND=100*((CRISIS_D.InvN ./ CRISIS_D.InvN[1]) .- 1)
    yyTP=100*((CRISIS_P.InvT ./ CRISIS_P.InvT[1]) .- 1)
    yyTD=100*((CRISIS_D.InvT ./ CRISIS_D.InvT[1]) .- 1)
    zz=zeros(length(yyTD))
    YLIMS=[minimum([yyNP;yyND;yyTP;yyTD]),maximum([yyNP;yyND;yyTP;yyTD])]
    #Plot investment in N
    TITLE="non-tradable investment"
    YLABEL="percentage change"
    plt_invN=plot(xx,[yyNP yyND],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 ylims=YLIMS,
                 label=["Planner" "Decentralized"],
                 legend=:bottomright,size=(size_width,size_height))

    #Plot investment in T
    TITLE="tradable investment"
    YLABEL="percentage change"
    plt_invT=plot(xx,[yyTP yyTD],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 ylims=YLIMS,
                 legend=false,size=(size_width,size_height))

    #Plot subsidy rates
    TITLE="change in fist-best subsidy rates"
    YLABEL="basis points"
    tauN=10000*CRISIS_D.TAU_N
    tauT=10000*CRISIS_D.TAU_T
    plt_tau=plot(xx,[tauN.-tauN[1] tauT.-tauT[1] zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=[:green :orange :black],
                 label=["τN" "τT" "0"],
                 legend=:best,size=(size_width,size_height))

    #Plot total cost of subsidies
    TITLE="total cost of subsidies"
    YLABEL="percentage of GDP"
    subN=CRISIS_D.TAU_N .* CRISIS_D.InvN
    subT=CRISIS_D.TAU_T .* CRISIS_D.InvT
    P=1 ./ CRISIS_D.RER
    gdp=P .* CRISIS_D.GDP .+ CRISIS_D.TB
    subCost=100*(P .* (subN+subT)) ./ gdp
    plt_SCost=plot(xx,subCost,xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=:solid,
                 linecolor=:purple,
                 ylims=[0.06,0.09],
                 legend=false,size=(size_width,size_height))
    #Create plot array
    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_invN,plt_invT,plt_tau,plt_SCost,
             layout=l,size=(size_width*2,size_height*2))
    FOLDER_GRAPHS="Plots"
    savefig(plt,"$FOLDER_GRAPHS\\CrisisPlotsSectors.pdf")
    return plt
end

###############################################################################
#Functions to create Impulse-Response functions
###############################################################################

function GetInitialState_ForIR(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, TsinceDefault, Tsim = par
    #Get initial state and path after TsinceDefault with no default
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
    NLongSamples=1
    tsince=0
    while true
        if t0+10==length(STATES_TS.Def)
            NLongSamples=NLongSamples+1
            if NLongSamples<=5
                #Get another long sample
                STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
                t0=drp+1
                tsince=0
            else
                println("failed to get T since default in good standing")
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
            #Found t0 of TsinceDefault in good standing
            break
        end
    end
    return STATES_TS.z[t0], STATES_TS.Def[t0], STATES_TS.KN[t0], STATES_TS.KT[t0], STATES_TS.B[t0]
end

function GetImpulseResponse_States(sd::Int64,TafterShock::Int64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack μ_z, ρ_z, σ_ϵz = par

    z_1, Def_1, kN_1, kT_1, b_1 = GetInitialState_ForIR(SOLUTION,GRIDS,par)
    #Get shocks, at t=0 make z drop 1 standard deviation
    z0=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_1)-1.0*σ_ϵz)
    Random.seed!(sd)
    zz, xx=Simulate_z_shocks_z0(z0,TafterShock+1,GRIDS,par)
    zTS=vcat(z_1,zz)

    ST=SimulateStates_GivenShocks(zTS,Def_1,kN_1,kT_1,b_1,SOLUTION,GRIDS,par)

    Tries=1
    while sum(ST.Def[1:end])>0.0
        z_1, Def_1, kN_1, kT_1, b_1 = GetInitialState_ForIR(SOLUTION,GRIDS,par)
        #Get shocks, at t=0 make z drop 1 standard deviation
        z0=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_1)-1.0*σ_ϵz)
        Random.seed!(sd+Tries)
        zz, xx=Simulate_z_shocks_z0(z0,TafterShock+1,GRIDS,par)
        zTS=vcat(z_1,zz)

        ST=SimulateStates_GivenShocks(zTS,Def_1,kN_1,kT_1,b_1,SOLUTION,GRIDS,par)
        Tries=Tries+1
        if Tries>10
            println("Failed to find IR without default episode")
            break
        end
    end

    return ST
end

function GetImpulseResponse_Path(sd::Int64,TafterShock::Int64,SOLUTION::Solution,SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    #Get path of states
    S_IR=GetImpulseResponse_States(sd,TafterShock,SOLUTION,GRIDS,par)
    TAU_N, TAU_T=OptimalSubsidiesSeries(S_IR,SOL_PLA,GRIDS,par)

    #Compute variables
    T=length(S_IR.z)
    #States
    z=S_IR.z
    Def=S_IR.Def
    KN=S_IR.KN
    KT=S_IR.KT
    B=S_IR.B
    #Variables
    Spreads=S_IR.Spreads
    GDP=Array{Float64,1}(undef,T)
    InvN=Array{Float64,1}(undef,T)
    InvT=Array{Float64,1}(undef,T)
    Cons=Array{Float64,1}(undef,T)
    TB=Array{Float64,1}(undef,T)
    CA=Array{Float64,1}(undef,T)
    RER=Array{Float64,1}(undef,T)

    @unpack itp_q1 = SOLUTION
    @unpack γ, κ, δ = par

    for t in 1:T
        if t==T
            @unpack itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
            kNprime=itp_kNprime(B[t],KT[t],KN[t],z[t])
            kTprime=itp_kTprime(B[t],KT[t],KN[t],z[t])
            bprime=itp_bprime(B[t],KT[t],KN[t],z[t])
        else
            kNprime=KN[t+1]
            kTprime=KT[t+1]
            bprime=B[t+1]
        end
        qq=itp_q1(bprime,kTprime,kNprime,z[t])
        Tr=qq*(bprime-(1-γ)*B[t])-(γ+(1-γ)*κ)*B[t]
        GDP[t]=FinalOutput(z[t],KN[t],KT[t],Tr,par)
        RER[t]=1.0/PriceFinalGood(z[t],KN[t],KT[t],Tr,par)
        # RER[t]=PriceNonTraded(z[t],KN[t],KT[t],Tr,par)
        InvN[t]=kNprime-(1-δ)*KN[t]
        InvT[t]=kTprime-(1-δ)*KT[t]
        Cons[t]=ConsNet(GDP[t],KN[t],KT[t],kNprime,kTprime,par)
        TB[t]=-Tr
        CA[t]=-(bprime-B[t])
    end
    return Paths(z,Def,KN,KT,B,Spreads,GDP,InvN,InvT,Cons,TB,CA,RER,TAU_N,TAU_T)
end

function AverageIRPaths(TafterShock::Int64,SOLUTION::Solution,SOL_PLA::Solution,GRIDS::Grids,par::Pars)
    @unpack N_crises = par
    Random.seed!(1234)
    #Do first
    IR=GetImpulseResponse_Path(1,TafterShock,SOLUTION,SOL_PLA,GRIDS,par)

    #Add paths
    for i=2:N_crises
        println("Doing IR path i=$i")
        C1=GetImpulseResponse_Path(i,TafterShock,SOLUTION,SOL_PLA,GRIDS,par)
        IR.z=IR.z .+ C1.z
        IR.Def=IR.Def .+ C1.Def
        IR.KN=IR.KN .+ C1.KN
        IR.KT=IR.KT .+ C1.KT
        IR.B=IR.B .+ C1.B
        #Variables
        IR.Spreads=IR.Spreads .+ C1.Spreads
        IR.GDP=IR.GDP .+ C1.GDP
        IR.InvN=IR.InvN .+ C1.InvN
        IR.InvT=IR.InvT .+ C1.InvT
        IR.Cons=IR.Cons .+ C1.Cons
        IR.TB=IR.TB .+ C1.TB
        IR.CA=IR.CA .+ C1.CA
        IR.RER=IR.RER .+ C1.RER
        IR.TAU_N=IR.TAU_N .+ C1.TAU_N
        IR.TAU_T=IR.TAU_T .+ C1.TAU_T
    end
    #Average
    IR.z=IR.z ./ N_crises
    IR.Def=IR.Def ./ N_crises
    IR.KN=IR.KN ./ N_crises
    IR.KT=IR.KT ./ N_crises
    IR.B=IR.B ./ N_crises
    #Variables
    IR.Spreads=IR.Spreads ./ N_crises
    IR.GDP=IR.GDP ./ N_crises
    IR.InvN=IR.InvN ./ N_crises
    IR.InvT=IR.InvT ./ N_crises
    IR.Cons=IR.Cons ./ N_crises
    IR.TB=IR.TB ./ N_crises
    IR.CA=IR.CA ./ N_crises
    IR.RER=IR.RER ./ N_crises
    IR.TAU_N=IR.TAU_N ./ N_crises
    IR.TAU_T=IR.TAU_T ./ N_crises
    return IR
end

function CreateTwo_IR_Plots(IR_P::Paths,IR_D::Paths,par::Pars)
    @unpack size_width, size_height = par
    #General objects for plots
    xx=[-1:1:length(IR_P.z)-2]
    LABELS=["Planner" "Decentralized" "0"]
    LINESTYLE=[:solid :dash :solid]
    LINECOLOR=[:blue :red :black]

    #Plot productivity shock
    TITLE="productivity shock: -1std(z)"
    YLABEL="percentage change"
    yyP=100*((IR_P.z./IR_P.z[1]).-1)
    yyD=100*((IR_D.z./IR_D.z[1]).-1)
    zz=zeros(Float64,length(yyP))
    plt_z=plot(xx,[yyP yyD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 label=LABELS,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 legend=:bottomright,size=(size_width,size_height))

    #Plot GDP
    TITLE="GDP"
    YLABEL="percentage change"
    # yyP=100*((IR_P.GDP./IR_P.GDP[1]).-1)
    # yyD=100*((IR_D.GDP./IR_D.GDP[1]).-1)
    pP=1 ./IR_P.RER
    pD=1 ./IR_D.RER
    gdpP=pP.*IR_P.GDP.+IR_P.TB
    gdpD=pD.*IR_D.GDP.+IR_D.TB
    yyP=100*((gdpP./gdpP[1]).-1)
    yyD=100*((gdpD./gdpD[1]).-1)
    plt_gdp=plot(xx,[yyP yyD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 legend=false,size=(size_width,size_height))

    #Plot Spreads
    TITLE="change in spreads"
    YLABEL="percentage points"
    yyP=IR_P.Spreads .- IR_P.Spreads[1]
    yyD=IR_D.Spreads .- IR_D.Spreads[1]
    plt_spreads=plot(xx,[yyP yyD zz],xlabel="t",
                     title=TITLE,ylabel=YLABEL,
                     linestyle=LINESTYLE,
                     linecolor=LINECOLOR,
                     legend=false,size=(size_width,size_height))

    #Plot debt
    TITLE="current account"
    YLABEL="percentage of AvGDP"
    yyP=100*IR_P.CA ./ IR_P.GDP[1]
    yyD=100*IR_D.CA ./ IR_D.GDP[1]
    plt_b=plot(xx,[yyP.-yyP[1] yyD.-yyD[1] zz],xlabel="t",
               title=TITLE,ylabel=YLABEL,
               linestyle=LINESTYLE,
               linecolor=LINECOLOR,
               legend=false,size=(size_width,size_height))

    #Plot investment
    TITLE="investment"
    YLABEL="percentage change"
    invP=IR_P.InvN + IR_P.InvT
    invD=IR_D.InvN + IR_D.InvT
    yyP=100*((invP ./ invP[1]) .- 1)
    yyD=100*((invD./invD[1]) .- 1)
    plt_inv=plot(xx,[yyP yyD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,
                 legend=false,size=(size_width,size_height))

    #Plot Consumption
    TITLE="consumption"
    YLABEL="percentage change"
    yyP=100*((IR_P.Cons./IR_P.Cons[1]).-1)
    yyD=100*((IR_D.Cons./IR_D.Cons[1]).-1)
    plt_con=plot(xx,[yyP yyD zz],xlabel="t",
               title=TITLE,ylabel=YLABEL,
               linestyle=LINESTYLE,
               linecolor=LINECOLOR,
               legend=false,size=(size_width,size_height))

    #Create plot array
    #Create plot array
    l = @layout([a b c; d e f])
    plt=plot(plt_z,plt_spreads,plt_b,plt_inv,plt_gdp,plt_con,
             layout=l,size=(size_width*3,size_height*2))
    FOLDER_GRAPHS="Plots"
    savefig(plt,"$FOLDER_GRAPHS\\IR_aggregates.pdf")
    return plt
end

function CreateTwo_IR_Plots_Sectors(IR_P::Paths,IR_D::Paths,par::Pars)
    @unpack size_width, size_height = par
    #General objects for plots
    xx=[-1:1:length(IR_P.z)-2]
    LABELS=["Planner" "Decentralized" "0"]
    LINESTYLE=[:solid :dash :solid]
    LINECOLOR=[:blue :red :black]

    #Plot RER
    TITLE="real exchange rate, 1/P"
    YLABEL="percentage change"
    yyP=100*((IR_P.RER./IR_P.RER[1]).-1)
    yyD=100*((IR_D.RER./IR_D.RER[1]).-1)
    zz=zeros(Float64,length(yyP))
    plt_rer=plot(xx,[yyP yyD zz],xlabel="t",
               title=TITLE,ylabel=YLABEL,
               label=LABELS,
               linestyle=LINESTYLE,
               linecolor=LINECOLOR,
               legend=:best,size=(size_width,size_height))

    #Plots investment
    yyNP=100*((IR_P.InvN ./ IR_P.InvN[1]) .- 1)
    yyND=100*((IR_D.InvN ./ IR_D.InvN[1]) .- 1)
    yyTP=100*((IR_P.InvT ./ IR_P.InvT[1]) .- 1)
    yyTD=100*((IR_D.InvT ./ IR_D.InvT[1]) .- 1)
    YLIMS=[minimum([yyNP;yyND;yyTP;yyTD]),maximum([yyNP;yyND;yyTP;yyTD])]
    #Plot investment in N
    TITLE="investment in non-tradable"
    YLABEL="percentage change"
    plt_invN=plot(xx,[yyNP yyND zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,ylims=YLIMS,
                 legend=false,size=(size_width,size_height))

    #Plot investment in T
    TITLE="investment in tradable"
    YLABEL="percentage change"
    plt_invT=plot(xx,[yyTP yyTD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,ylims=YLIMS,
                 legend=false,size=(size_width,size_height))

    #Plot consumption of N and T
    yN_P=Array{Float64,1}(undef,length(IR_P.InvN))
    yN_D=Array{Float64,1}(undef,length(IR_P.InvN))
    yT_P=Array{Float64,1}(undef,length(IR_P.InvN))
    yT_D=Array{Float64,1}(undef,length(IR_P.InvN))
    for t=1:length(IR_P.z)
        yN_P[t]=NonTradedProduction(IR_P.z[t],IR_P.KN[t],par)
        yN_D[t]=NonTradedProduction(IR_D.z[t],IR_D.KN[t],par)
    end
    yyNP=100*((yN_P ./ yN_P[1]) .- 1)
    yyND=100*((yN_D ./ yN_D[1]) .- 1)
    for t=1:length(IR_P.z)
        yT_P[t]=TradedProduction(IR_P.z[t],IR_P.KT[t],par)-IR_P.TB[t]
        yT_D[t]=TradedProduction(IR_D.z[t],IR_D.KT[t],par)-IR_D.TB[t]
    end
    yyTP=100*((yT_P ./ yT_P[1]) .- 1)
    yyTD=100*((yT_D ./ yT_D[1]) .- 1)
    YLIMS=[minimum([yyNP;yyND;yyTP;yyTD]),maximum([yyNP;yyND;yyTP;yyTD])]

    #Plot consumption of N
    TITLE="consumption of non-tradable"
    YLABEL="percentage change"
    plt_cN=plot(xx,[yyNP yyND zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,ylims=YLIMS,
                 legend=false,size=(size_width,size_height))

    #Plot production of T
    TITLE="consumption of tradable"
    YLABEL="percentage change"
    plt_cT=plot(xx,[yyTP yyTD zz],xlabel="t",
                 title=TITLE,ylabel=YLABEL,
                 linestyle=LINESTYLE,
                 linecolor=LINECOLOR,ylims=YLIMS,
                 legend=false,size=(size_width,size_height))

    #Plot TAU
    TITLE="change in fist-best subsidy rates"
    YLABEL="basis points"
    yyN=10000*(IR_D.TAU_N .- IR_D.TAU_N[1])
    yyT=10000*(IR_D.TAU_T .- IR_D.TAU_T[1])
    LABELS_T=["τN" "τT" "0"]
    LINECOLOR_T=[:green :orange :black]
    plt_tau=plot(xx,[yyN,yyT,zeros(Float64,length(yyN))],xlabel="t",label=LABELS_T,
                title=TITLE,ylabel=YLABEL,
                linestyle=[:solid :dash :solid],
                linecolor=LINECOLOR_T,
                legend=:best,size=(size_width,size_height))

    #Plot cost of subsidy
    TITLE="cost of optimal subsidies"
    YLABEL="percentage of AvGDP"
    yyN=100*(IR_D.TAU_N .* IR_D.InvN ./ IR_D.GDP[1])
    yyT=100*(IR_D.TAU_T .* IR_D.InvT ./ IR_D.GDP[1])
    LABELS_T=["τN" "τT"]
    LINECOLOR_T=[:green :orange]
    plt_tau_cost=plot(xx,[yyN.-yyN[1],yyT.-yyT[1]],xlabel="t",label=LABELS_T,
                title=TITLE,ylabel=YLABEL,
                linestyle=[:solid :dash],
                linecolor=LINECOLOR_T,
                legend=:best,size=(size_width,size_height))

    #Plot total cost of subsidies
    TITLE="total cost of optimal subsidies"
    YLABEL="percentage of AvGDP"
    yy=100*(((IR_D.TAU_N .* IR_D.InvN) .+ (IR_D.TAU_T .* IR_D.InvT)) ./ IR_D.GDP[1])
    plt_tau_total=plot(xx,yy,xlabel="t",
                title=TITLE,ylabel=YLABEL,
                linestyle=:solid,
                linecolor=:black,
                legend=false,size=(size_width,size_height))

    #Create plot array
    #Create plot array
    l = @layout([a b c; d e f])
    FOLDER_GRAPHS="Plots"
    plt=plot(plt_invN,plt_invT,plt_rer,plt_cN,plt_cT,plt_tau,
             layout=l,size=(size_width*3,size_height*2))
    savefig(plt,"$FOLDER_GRAPHS\\IR_Sectors.pdf")
    return plt
end

###############################################################################
#Functions to get path for table
###############################################################################
function GetPathsFromStates(S_path::States_TS,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Get path of states
    TAU_N, TAU_T=OptimalSubsidiesSeries(S_path,SOLUTION,GRIDS,par)

    #Compute variables
    T=length(S_path.z)
    #States
    z=S_path.z
    Def=S_path.Def
    KN=S_path.KN
    KT=S_path.KT
    B=S_path.B
    #Variables
    Spreads=S_path.Spreads
    GDP=Array{Float64,1}(undef,T)
    InvN=Array{Float64,1}(undef,T)
    InvT=Array{Float64,1}(undef,T)
    Cons=Array{Float64,1}(undef,T)
    TB=Array{Float64,1}(undef,T)
    CA=Array{Float64,1}(undef,T)
    RER=Array{Float64,1}(undef,T)

    @unpack itp_q1 = SOLUTION
    @unpack γ, κ, δ = par

    for t in 1:T
        if t==T
            if S_path.Def[t]==0
                @unpack itp_kNprime, itp_kTprime, itp_bprime = SOLUTION
                kNprime=itp_kNprime(B[t],KT[t],KN[t],z[t])
                kTprime=itp_kTprime(B[t],KT[t],KN[t],z[t])
                bprime=itp_bprime(B[t],KT[t],KN[t],z[t])
            else
                @unpack itp_kNprime_D, itp_kTprime_D = SOLUTION
                kNprime=itp_kNprime_D(KT[t],KN[t],z[t])
                kTprime=itp_kTprime_D(KT[t],KN[t],z[t])
                bprime=0.0
            end
        else
            kNprime=KN[t+1]
            kTprime=KT[t+1]
            bprime=B[t+1]
        end
        if S_path.Def[t]==0
            z_at=z[t]
            qq=itp_q1(bprime,kTprime,kNprime,z[t])
            Tr=qq*(bprime-(1-γ)*B[t])-(γ+(1-γ)*κ)*B[t]
        else
            z_at=zDefault(z[t],par)
            Tr=0.0
            TAU_N[t]=0.0
            TAU_T[t]=0.0
        end
        P=PriceFinalGood(z_at,KN[t],KT[t],Tr,par)
        Y=FinalOutput(z_at,KN[t],KT[t],Tr,par)
        GDP[t]=P*Y-Tr
        RER[t]=1.0/P
        InvN[t]=kNprime-(1-δ)*KN[t]
        InvT[t]=kTprime-(1-δ)*KT[t]
        Cons[t]=ConsNet(Y,KN[t],KT[t],kNprime,kTprime,par)
        TB[t]=-Tr
        CA[t]=-(bprime-B[t])
    end
    return Paths(z,Def,KN,KT,B,Spreads,GDP,InvN,InvT,Cons,TB,CA,RER,TAU_N,TAU_T)
end
