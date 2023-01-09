

include("Primitives.jl")

β=0.9756005859375
φ=2.617919921875
d1=0.26619873046875
knk=0.74178466796875
d0=-knk*d1
par, GRIDS = Setup(β,φ,d0,d1)
SolveAndSaveModel_VFI(GRIDS,par)
