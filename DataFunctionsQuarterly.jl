using DataFrames, CSV, Statistics

function PlotRealExchangeRate_q(df::DataFrame,y0::Int64,yT::Int64,
                              country::Array{String},
                              countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0-1))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    XTICKS=string.(dfCountry[:,:year][5:end])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    x=dfCountry[:,:Quarter]
    xx=x[5:end]
    rer=dfCountry[:,:reer]
    yy=Array{Float64,1}(undef,length(xx))
    for i in 5:length(x)
        yy[i-4]=mean(rer[i-3:i])
    end
    plt=plot(xx,100 .*(yy./yy[1]).-100,label=countryLabels[1],
             legend=false,title="real exchange rate",
             linestyle=LINESTYLES[1],
             color=COLORS[1],
             ylabel="percentage change",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0-1))
        #Add to plot
        rer=dfCountry[:,:reer]
        yy=Array{Float64,1}(undef,length(xx))
        for i in 5:length(x)
            yy[i-4]=mean(rer[i-3:i])
        end
        if m==0
            plt=plot!(xx,100 .*(yy./yy[1]).-100,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,100 .*(yy./yy[1]).-100,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],
                      markershape=MARKERS[m],
                      markerstrokecolor=:match)
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotSpreads_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do Germany
    dfGER=subset(df,Pair(:country,c -> c.=="GERMANY"),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[0.0,10.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yy=dfCountry[:,:int_rate] .- dfGER[:,:int_rate]
    plt=plot(xx,yy,label=countryLabels[1],
             ylims=YLIMS,legend=:topright,
             linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="Spreads on 10-year sovereign bonds",
             ylabel="percent per year",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yy=dfCountry[:,:int_rate] .- dfGER[:,:int_rate]
        if m==0
            plt=plot!(xx,yy,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotGDP_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yyy=dfCountry[:,:GDP_real]
    yy=100*((yyy./yyy[1]).-1)
    plt=plot(xx,yy,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="GDP",
             ylabel="percentage change",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yyy=dfCountry[:,:GDP_real]
        yy=100*((yyy./yyy[1]).-1)
        if m==0
            plt=plot!(xx,yy,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy .- yy[1],label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotConsumption_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yyy=dfCountry[:,:Consumption]
    yy=100*((yyy./yyy[1]).-1)
    plt=plot(xx,yy,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="consumption",
             ylabel="percentage change",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yyy=dfCountry[:,:Consumption]
        yy=100*((yyy./yyy[1]).-1)
        if m==0
            plt=plot!(xx,yy,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy .- yy[1],label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotCummCA_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yyy=dfCountry[:,:CurrentAccount_GDP]
    yy=zeros(Float64,length(yyy))
    for t=2:length(yyy)
        yy[t]=yy[t-1]+yyy[t]
    end
    plt=plot(xx,-yy/4,label=countryLabels[1],#ylims=YLIMS,
             legend=:topleft,linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="cummulative CA deficits (Δb)",
             ylabel="percentage of GDP",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yyy=dfCountry[:,:CurrentAccount_GDP]
        yy=zeros(Float64,length(yyy))
        for t=2:length(yyy)
            yy[t]=yy[t-1]+yyy[t]
        end
        if m==0
            plt=plot!(xx,-yy/4,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy/4,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotCummTB_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yyy=dfCountry[:,:TradeBalance_GDP]
    yy=zeros(Float64,length(yyy))
    for t=2:length(yyy)
        yy[t]=yy[t-1]+yyy[t]
    end
    plt=plot(xx,yy/4,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="cummulative trade balance",
             ylabel="percentage of GDP",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yyy=dfCountry[:,:TradeBalance_GDP]
        yy=zeros(Float64,length(yyy))
        for t=2:length(yyy)
            yy[t]=yy[t-1]+yyy[t]
        end
        if m==0
            plt=plot!(xx,yy/4,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy/4,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotTB_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yy=dfCountry[:,:TradeBalance_GDP]
    plt=plot(xx,[yy.-yy[1]]./4,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="trade balance",
             ylabel="percentage of GDP",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yy=dfCountry[:,:TradeBalance_GDP]
        if m==0
            plt=plot!(xx,[yy.-yy[1]]./4,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy/4,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end

function PlotInvestment_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot, :solid]
    COLORS=[:blue, :red, :green, :orange, :black]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    yyy=dfCountry[:,:Investment]
    yy=100*((yyy./yyy[1]).-1)
    plt=plot(xx,yy,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             color=COLORS[1],
             title="Investment",
             ylabel="percentage change",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        yyy=dfCountry[:,:Investment]
        yy=100*((yyy./yyy[1]).-1)
        if m==0
            plt=plot!(xx,yy,label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l])
        else
            plt=plot!(xx,yy .- yy[1],label=countryLabels[i],
                      linestyle=LINESTYLES[l],
                      color=COLORS[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    zz=zeros(Float64,length(yy))
    plt=plot!(xx,zz,label="0",
              linestyle=LINESTYLES[end],
              color=COLORS[end])
    return plt
end



function PlotVAsectorT_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    va_A=dfCountry[:,:VA_A]
    va_B_E=dfCountry[:,:VA_B_E]
    va_C=dfCountry[:,:VA_C]
    va_F=dfCountry[:,:VA_F]
    va_G_I=dfCountry[:,:VA_G_I]
    va_J=dfCountry[:,:VA_J]
    va_K=dfCountry[:,:VA_K]
    va_L=dfCountry[:,:VA_L]
    va_M_N=dfCountry[:,:VA_M_N]
    va_O_Q=dfCountry[:,:VA_O_Q]
    va_R_U=dfCountry[:,:VA_R_U]

    yy_T=va_A.+va_B_E.+va_C
    yy_N=va_F.+va_G_I.+va_J.+va_K.+va_L.+va_M_N.+va_O_Q.+va_R_U

    yyT=100*((yy_T./yy_T[1]).-1)
    yyN=100*((yy_N./yy_N[1]).-1)
    plt=plot(xx,yyT,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             title="Value Added, Traded Sectors",
             ylabel="percentage change",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        va_A=dfCountry[:,:VA_A]
        va_B_E=dfCountry[:,:VA_B_E]
        va_C=dfCountry[:,:VA_C]
        va_F=dfCountry[:,:VA_F]
        va_G_I=dfCountry[:,:VA_G_I]
        va_J=dfCountry[:,:VA_J]
        va_K=dfCountry[:,:VA_K]
        va_L=dfCountry[:,:VA_L]
        va_M_N=dfCountry[:,:VA_M_N]
        va_O_Q=dfCountry[:,:VA_O_Q]
        va_R_U=dfCountry[:,:VA_R_U]

        yy_T=va_A.+va_B_E.+va_C
        yy_N=va_F.+va_G_I.+va_J.+va_K.+va_L.+va_M_N.+va_O_Q.+va_R_U

        yyT=100*((yy_T./yy_T[1]).-1)
        yyN=100*((yy_N./yy_N[1]).-1)
        if m==0
            plt=plot!(xx,yyT,label=countryLabels[i],
                      linestyle=LINESTYLES[l])
        else
            plt=plot!(xx,yyT .- yyT[1],label=countryLabels[i],
                      linestyle=LINESTYLES[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    return plt
end

function PlotVAsectorN_q(df::DataFrame,y0::Int64,yT::Int64,
                           country::Array{String},
                           countryLabels::Array{String})
    #Do first
    dfCountry=subset(df,Pair(:country,c -> c.==country[1]),
                     Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
    #Create plot
    LINESTYLES=[:solid, :dash, :dot, :dashdot]
    MARKERS=[:circle, :diamond, :star5, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x]
    YLIMS=[2.0,16.0]
    XTICKS=string.(dfCountry[:,:year])
    for i in 1:length(XTICKS)
        if mod(i+3,4)>0
            XTICKS[i]=""
        end
    end
    xx=dfCountry[:,:Quarter]
    va_A=dfCountry[:,:VA_A]
    va_B_E=dfCountry[:,:VA_B_E]
    va_C=dfCountry[:,:VA_C]
    va_F=dfCountry[:,:VA_F]
    va_G_I=dfCountry[:,:VA_G_I]
    va_J=dfCountry[:,:VA_J]
    va_K=dfCountry[:,:VA_K]
    va_L=dfCountry[:,:VA_L]
    va_M_N=dfCountry[:,:VA_M_N]
    va_O_Q=dfCountry[:,:VA_O_Q]
    va_R_U=dfCountry[:,:VA_R_U]

    yy_T=va_A.+va_B_E.+va_C
    yy_N=va_F.+va_G_I.+va_J.+va_K.+va_L.+va_M_N.+va_O_Q.+va_R_U

    yyT=100*((yy_T./yy_T[1]).-1)
    yyN=100*((yy_N./yy_N[1]).-1)
    plt=plot(xx,yyN,label=countryLabels[1],#ylims=YLIMS,
             legend=false,linestyle=LINESTYLES[1],
             title="Value Added, Non-traded Sectors",
             ylabel="percentage change",xticks=(1:1:length(xx),XTICKS))

    #Loop over country codes and labels
    i=1
    l=2
    m=0
    while i<length(country)
        i=i+1
        #Extract df for country
        dfCountry=subset(df,Pair(:country,c -> c.==country[i]),
                         Pair(:year,y -> y.<=yT),Pair(:year,y -> y.>=y0))
        #Add to plot
        va_A=dfCountry[:,:VA_A]
        va_B_E=dfCountry[:,:VA_B_E]
        va_C=dfCountry[:,:VA_C]
        va_F=dfCountry[:,:VA_F]
        va_G_I=dfCountry[:,:VA_G_I]
        va_J=dfCountry[:,:VA_J]
        va_K=dfCountry[:,:VA_K]
        va_L=dfCountry[:,:VA_L]
        va_M_N=dfCountry[:,:VA_M_N]
        va_O_Q=dfCountry[:,:VA_O_Q]
        va_R_U=dfCountry[:,:VA_R_U]

        yy_T=va_A.+va_B_E.+va_C
        yy_N=va_F.+va_G_I.+va_J.+va_K.+va_L.+va_M_N.+va_O_Q.+va_R_U

        yyT=100*((yy_T./yy_T[1]).-1)
        yyN=100*((yy_N./yy_N[1]).-1)
        if m==0
            plt=plot!(xx,yyN,label=countryLabels[i],
                      linestyle=LINESTYLES[l])
        else
            plt=plot!(xx,yyN .- yyT[1],label=countryLabels[i],
                      linestyle=LINESTYLES[l],markershape=MARKERS[m])
        end
        l=l+1
        if l>length(LINESTYLES)
            l=1
            m=m+1
        end
    end
    return plt
end
