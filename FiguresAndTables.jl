
using Distributed, BenchmarkTools
using Plots; pyplot(fontfamily="serif",linewidth=2.5,grid=false,legend=true,
                    background_color_legend=nothing,foreground_color_legend=nothing,
                    legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                    markersize=9,size=(650,500))


################################################################################
########## Load primitives and solutions #######################################
################################################################################
include("Primitives.jl")
β=0.9756005859375
φ=2.617919921875
d1=0.26619873046875
knk=0.74178466796875
d0=-knk*d1
par, GRIDS = Setup(β,φ,d0,d1)

SOL_PLA=Unpack_Solution("Planner",GRIDS,par)
SOL_DEC=Unpack_Solution("Decentralized",GRIDS,par)

################################################################################
########### Compute moments over business cycle ################################
################################################################################
#Moments for Table 1 come from here
MOM_PLA=AverageMomentsManySamples(SOL_PLA,GRIDS,par)
MOM_DEC=AverageMomentsManySamples(SOL_DEC,GRIDS,par)

################################################################################
######################### Figure 1, price q(x',z) ##############################
################################################################################
par=Pars(par,Tsim=10000)
SP=SimulateStates_Long(SOL_PLA,GRIDS,par)
LINESTYLE=[:solid :dash :dot]
LINECOLOR=[:blue :red :green]
bm=mean(SP.B)
bl=bm-2.0*std(SP.B)
bh=bm+2.0*std(SP.B)
nn=1000
z=1.0
YLIMS=[0.5,1.3]
#Plot q as a function of KN, fixed KT, different B
KT=mean(SP.KT)
xx=collect(range(par.kNlow,stop=par.kNhigh,length=nn))
bprimeL=round(bl,digits=2)
fooL(x::Float64)=SOL_PLA.itp_q1(bprimeL,KT,x,z)
bprimeM=round(bm,digits=2)
fooM(x::Float64)=SOL_PLA.itp_q1(bprimeM,KT,x,z)
bprimeH=round(bh,digits=2)
fooH(x::Float64)=SOL_PLA.itp_q1(bprimeH,KT,x,z)
KTrnd=round(KT,digits=1)
plt_qkN=plot(xx,1*[fooL.(xx) fooM.(xx) fooH.(xx)],
     label=["Av(B')-2std" "Av(B')" "Av(B')+2std"],
     ylabel="q(x',z)",title="Price q, z=1, KT'=Av(KT)",
     legend=false,linestyle=LINESTYLE,linecolor=LINECOLOR,
     xlabel="KN'",ylims=YLIMS)

#Plot q as a function of KT, fixed KN, different B
KN=mean(SP.KN)
xx=collect(range(par.kTlow,stop=par.kThigh,length=nn))
bprimeL=round(bl,digits=2)
fooL(x::Float64)=SOL_PLA.itp_q1(bprimeL,x,KN,z)
bprimeM=round(bm,digits=2)
fooM(x::Float64)=SOL_PLA.itp_q1(bprimeM,x,KN,z)
bprimeH=round(bh,digits=2)
fooH(x::Float64)=SOL_PLA.itp_q1(bprimeH,x,KN,z)
KNrnd=round(KN,digits=1)
plt_qkT=plot(xx,1*[fooL.(xx) fooM.(xx) fooH.(xx)],
     label=["Av(B')-2std" "Av(B')" "Av(B')+2std"],
     ylabel="q(x',z)",title="Price q, z=1, KN'=Av(KN)",
     legend=:bottomright,linestyle=LINESTYLE,linecolor=LINECOLOR,
     xlabel="KT'",ylims=YLIMS)

l = @layout([a b])
FOLDER_GRAPHS="Plots"
plt=plot(plt_qkN,plt_qkT,
         layout=l,size=(par.size_width*2,par.size_height))
savefig(plt,"Plots\\Price_q_kN_kT.pdf")
################################################################################
############## Figure 2, price q as a function of Λ and K ######################
################################################################################
par=Pars(par,Tsim=10000)
SP=SimulateStates_Long(SOL_PLA,GRIDS,par)
LINESTYLE=[:solid :dash :dot]
LINECOLOR=[:blue :red :green]
#Plot q as a function of Λ, fixed K
bm=mean(SP.B)
bl=bm-2.0*std(SP.B)
bh=bm+2.0*std(SP.B)
K=mean(SP.KT+SP.KN)
nn=1000
z=1.0
xx=collect(range(0.15,stop=0.6,length=nn))
bprimeL=round(bl,digits=2)
fooL(x::Float64)=SOL_PLA.itp_q1(bprimeL,x*K,(1.0-x)*K,z)
bprimeM=round(bm,digits=2)
fooM(x::Float64)=SOL_PLA.itp_q1(bprimeM,x*K,(1.0-x)*K,z)
bprimeH=round(bh,digits=2)
fooH(x::Float64)=SOL_PLA.itp_q1(bprimeH,x*K,(1.0-x)*K,z)
Krnd=round(K,digits=1)
plt_qΛ=plot(xx,1*[fooL.(xx) fooM.(xx) fooH.(xx)],
     label=["Av(B')-2std" "Av(B')" "Av(B')+2std"],
     ylabel="q(x',z)",title="Price q, z=1, K=Av(KN+KT)",
     legend=:bottomright,linestyle=LINESTYLE,linecolor=LINECOLOR,
     xlabel="share of capital in tradable sector (Λ')")

#Plot q as a function of K, fixed Λ
Kbar=mean(SP.KN+SP.KT)
Klow=6#Kbar-2*std(SP.KN+SP.KT)
Khigh=13#Kbar+2*std(SP.KN+SP.KT)
Λbar=MOM_PLA.AvλK
nn=1000
z=1.0
xx=collect(range(Klow,stop=Khigh,length=nn))
bprimeL=round(bl,digits=2)
fooL(x::Float64)=SOL_PLA.itp_q1(bprimeL,Λbar*x,(1.0-Λbar)*x,z)
bprimeM=round(bm,digits=2)
fooM(x::Float64)=SOL_PLA.itp_q1(bprimeM,Λbar*x,(1.0-Λbar)*x,z)
bprimeH=round(bh,digits=2)
fooH(x::Float64)=SOL_PLA.itp_q1(bprimeH,Λbar*x,(1.0-Λbar)*x,z)
plt_qK=plot(xx,1*[fooL.(xx) fooM.(xx) fooH.(xx)],
     label=["Av(B')-1std" "Av(B')" "Av(B')+1std"],
     ylabel="q(x',z)",title="Price q, z=1 Λ'=Av(Λ)",
     legend=false,linestyle=LINESTYLE,linecolor=LINECOLOR,
     xlabel="total capital stock K'=KN'+KT'")#,ylims=[0.0,3.0])

l = @layout([a b])
FOLDER_GRAPHS="Plots"
plt=plot(plt_qK,plt_qΛ,
         layout=l,size=(par.size_width*2,par.size_height))
savefig(plt,"Plots\\Price_q_Theorems.pdf")

################################################################################
########### Long time series and moments for Table 2 ###########################
################################################################################
par=Pars(par,Tsim=11000)
SP=SimulateStates_Long(SOL_PLA,GRIDS,par)
SD=SimulateStates_Long(SOL_DEC,GRIDS,par)
PATH_P=GetPathsFromStates(SP,SOL_PLA,GRIDS,par)
PATH_D=GetPathsFromStates(SD,SOL_DEC,GRIDS,par)

t1=par.drp+1
tT=par.Tsim

#Column (1), total K
mean((PATH_D.KN[t1:tT] .+ PATH_D.KT[t1:tT]))
mean((PATH_P.KN[t1:tT] .+ PATH_P.KT[t1:tT]))

#Column (2), total K over GDP
mean((PATH_D.KN[t1:tT] .+ PATH_D.KT[t1:tT]) ./ PATH_D.GDP[t1:tT] ./ PATH_D.RER[t1:tT])
mean((PATH_P.KN[t1:tT] .+ PATH_P.KT[t1:tT]) ./ PATH_P.GDP[t1:tT] ./ PATH_P.RER[t1:tT])

#Column (3), Λ
mean(PATH_D.KT[t1:tT]./(PATH_D.KN[t1:tT] .+ PATH_D.KT[t1:tT]))
mean(PATH_P.KT[t1:tT]./(PATH_P.KN[t1:tT] .+ PATH_P.KT[t1:tT]))

#Column (4), total debt
sum(SD.B[t1:tT] .* (1 .- SD.Def[t1:tT]))/sum((1 .- SD.Def[t1:tT]))
sum(SP.B[t1:tT] .* (1 .- SP.Def[t1:tT]))/sum((1 .- SP.Def[t1:tT]))

#Column (5), debt-to-gdp ratio
#Check MOM_PLA and MOM_DEC above

#Column (6), total consumption
mean(PATH_D.Cons[t1:tT])
mean(PATH_P.Cons[t1:tT])

#Column (7), real exchange rate
mean(PATH_D.RER[t1:tT])
mean(PATH_P.RER[t1:tT])

mean(100*(log.(PATH_D.RER[t1:tT]).-log.(mean(PATH_P.RER[t1:tT]))))

100*std(log.(PATH_D.RER[t1:tT]) .- log(mean(PATH_D.RER[t1:tT])))
100*std(log.(PATH_P.RER[t1:tT]) .- log(mean(PATH_P.RER[t1:tT])))

################################################################################
########### Compute moments for subsidies and cost #############################
################################################################################
include("Primitives.jl")
par=Pars(par,Tsim=10000)
MOM_TAU=AverageTauMomentsManySamples(SOL_PLA,SOL_DEC,GRIDS,par)

################################################################################
########### Plot impulse-response functions to shocks ##########################
################################################################################
include("PlotsCrisis.jl")
par=Pars(par,N_crises=10000)
TafterShock=40
IR_P=AverageIRPaths(TafterShock,SOL_PLA,SOL_PLA,GRIDS,par)
IR_D=AverageIRPaths(TafterShock,SOL_DEC,SOL_PLA,GRIDS,par)
CreateTwo_IR_Plots(IR_P,IR_D,par)

CreateTwo_IR_Plots_Sectors(IR_P,IR_D,par)


################################################################################
#################### Plot data European debt crisis ############################
################################################################################

include("DataFunctionsQuarterly.jl")
df_q=DataFrame(CSV.File("quarterly_data.csv"))
y0=2009
yT=2015
country=["ITALY" "SPAIN" "PORTUGAL"]# "GREECE"]# "IRELAND"]
countryLabels=["Italy" "Spain" "Portugal"]# "Greece"]# "Ireland"]

plt_spr=PlotSpreads_q(df_q,y0,yT,country,countryLabels)
plt_gdp=PlotGDP_q(df_q,y0,yT,country,countryLabels)
plt_rer=PlotRealExchangeRate_q(df_q,y0,yT,country,countryLabels)
plt_con=PlotConsumption_q(df_q,y0,yT,country,countryLabels)
plt_tb=PlotTB_q(df_q,y0,yT,country,countryLabels)
plt_inv=PlotInvestment_q(df_q,y0,yT,country,countryLabels)

#Create plot array
l = @layout([a b c; d e f])
FOLDER_GRAPHS="Plots"
plt=plot(plt_spr,plt_tb,plt_rer,plt_inv,plt_gdp,plt_con,
         layout=l,size=(par.size_width*3,par.size_height*2))
savefig(plt,"$FOLDER_GRAPHS\\DataEuroCrisis.pdf")

################################################################################
######################### Plot model debt crisis ###############################
################################################################################

include("PlotsCrisis.jl")
par=Pars(par,N_crises=1500,t_before_crisis=10,t_after_crisis=18)
ΔSpr_DEC=3*MOM_DEC.StdSpreads
ΔSpr_PLA=3*MOM_PLA.StdSpreads
CRISIS_D, CRISIS_P=AverageBothCrisiesPaths(MOM_DEC.MeanSpreads+ΔSpr_DEC,SOL_PLA,SOL_DEC,GRIDS,par)
plt_C=CreateTwoCrisisPlots(CRISIS_P,CRISIS_D,par)

include("PlotsCrisis.jl")
plt_C2=CreateTwoCrisisPlotsSectors(CRISIS_P,CRISIS_D,par)
