source('./scripts/CVM_calc_Utilities.R')

args = commandArgs(trailingOnly = T)
id_file = as.character(args[1])

oripath = getwd()
tgt = paste('./Results/BuildNet/', id_file,'/wt-GDVs-4-A', sep = '')
setwd(tgt)

FeatureRAW.UnWeighted.GO = feature.Extall('', protein.info.graphletorbit, sum2)
FeatureMat.UnWeighted.GO = NetFeature(FeatureRAW.UnWeighted.GO)

FeatureRAW.Weighted.GO.CVM1 = feature.Extall.nPar('', protein.info.weighted.GOnPar, Cramer.von.dist, NoValue = 1)
FeatureMat.Weighted.GO.CVM1 = NetFeature(FeatureRAW.Weighted.GO.CVM1)

protein.labels = labelExType('')

setwd(oripath)
save.target = paste('./Results/ExtInfobyBatch/', id_file , sep = '')
dir.create(save.target, recursive = T)

save(FeatureMat.UnWeighted.GO, FeatureMat.Weighted.GO.CVM1, file = paste(save.target, '/', 'FeatureMat.RData',sep = ''))
save(protein.labels, file = paste(save.target, '/', 'labels.RData',sep = ''))