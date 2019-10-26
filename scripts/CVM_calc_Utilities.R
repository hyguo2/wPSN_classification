protein.info.graphlet35 = function(proteinfoldernm, func, type)
{
  summarytable = read.table(file = paste(proteinfoldernm, '/ew', sep = ''), header = F)
  Outputtable = summarytable[,-1]
  return(Outputtable)
}

protein.info.graphletorbit = function(proteinfoldernm, func, type)
{
  summarytable = read.table(file = paste(proteinfoldernm, '/ew.ecounts', sep = ''), header = F)
  Outputtable = summarytable[,-(1:3)]
  return(Outputtable)
}

proteinEx2b = function(proteinfoldernm, func, type)
{
  flist = list.files(proteinfoldernm)
  ll = length(flist)

  tableExt = function(filedir, func)
  {
    con = file(paste(proteinfoldernm, '/', filedir, sep = ''), open = 'r')
    Outlist = list()
    idx = 1
    while (length(oneLine <- readLines(con, n = 1, warn = FALSE)) > 0) 
    {
      OutVectorPre <- unlist(strsplit(oneLine, " "))
      if((lO <- length(OutVectorPre))<3) Outlist[[idx]] = 0
      else 
      {
        Outlist[[idx]] <- func(OutVectorPre[3:lO])
      }
      idx = idx + 1
    }
    close(con)
    return(unlist(Outlist))
  }
  
  Outputlist = lapply(seq(5, ll), function(x)tableExt(flist[x], func))
  
  Outputtable = matrix(unlist(Outputlist), ncol = ll-4)
  
  return(Outputtable)
}

protein.info.weighted.GOnPar = function(proteinfoldernm, func, type='edge', NoValue = 1)
{
  if(type == 'edge'){
    bound = 3
  }else if(type == 'node'){
    bound = 2
  }
  flist = list.files(proteinfoldernm)
  flist = flist[!grepl('CVM', flist)]
  ll = length(flist)
  
  basetb = read.table(paste(proteinfoldernm, '/', flist[4], sep = ''), header = F)
  basedist = basetb[,3]

  edgelabels = basetb[,c(1:2)]
  
  tableExt = function(filedir, func, obid)
  {
    con = file(paste(proteinfoldernm, '/', filedir, sep = ''), open = 'r')
    Outlist = list()
    idx = 1
    while (length(oneLine <- readLines(con, n = 1, warn = FALSE)) > 0) 
    {
      OutVectorPre <- unlist(strsplit(oneLine, " "))
      if((lO <- length(OutVectorPre))<bound) Outlist[[idx]] = NoValue #need to decide whether use 1 or 0
      else Outlist[[idx]] <- npardist2(func, OutVectorPre[bound:lO], obid)
      idx = idx + 1
    }
    close(con)
    return(unlist(Outlist))
  }
  
  Outputlist = lapply(seq(5, ll), function(x)tableExt(flist[x], func, basedist))
  
  Outputtable = matrix(unlist(Outputlist), ncol = ll-4)
  Outputtable = cbind(edgelabels, Outputtable)
  write.table(Outputtable, file = paste(proteinfoldernm, '/CVM1', sep = ''), row.names = F, col.names = F, quote = F)
  
  return(Outputtable)
}

feature.Extall = function(dirnm, func, func2, type = 'unknown')
{
  dirlist = list.dirs(path = paste('./', dirnm, sep = ''))[-1]
  dirlist = Cpcheck(dirlist)
  prolist = lapply(seq(1, length(dirlist)), function(x)func(dirlist[x], func2, type))
  return(prolist)
}

feature.Extall.nPar = function(dirnm, func, func2, type = 'edge', NoValue = 1)
{
  dirlist = list.dirs(path = paste('./', dirnm, sep = ''))[-1]
  dirlist = Cpcheck(dirlist)
  prolist = lapply(seq(1, length(dirlist)), function(x)func(dirlist[x], func2, type, NoValue))
  return(prolist)
}

feature.Extall.Base = function(dirnm, func, type = 'unknown')
{
  dirlist = list.dirs(path = paste('./', dirnm, sep = ''))[-1]
  dirlist = Cpcheck(dirlist)
  prolist = sapply(seq(1, length(dirlist)), function(x)func(dirlist[x]))
  return(t(prolist))
}

inputprocess = function(datavec)
{
  inputsplit = function(x)
  {
    Out = as.numeric(unlist(strsplit(x,':')))
    return(Out)
  }
  rt = sapply(datavec, inputsplit)
  return(rt)
}

sum2 = function(datavec)
{
  processedvec = inputprocess(datavec)
  return(sum(processedvec[1,]*processedvec[2,]))
}

sum2m = function(datavec, obid)
{
  processedvec = inputprocess(datavec)
  return(sum(processedvec[1,]*processedvec[2,])/obid)
}

mean2 = function(datavec)
{
  processedvec = inputprocess(datavec)
  return(sum(processedvec[1,]*processedvec[2,])/sum(processedvec[2,]))
}

geomean2 = function(datavec)
{
  processedvec = inputprocess(datavec)
  processedvec[1,] = log(processedvec[1,])
  gm = exp(sum(processedvec[1,]*processedvec[2,])/sum(processedvec[2,]))
  return(gm)
}

Cramer.von.dist = function(x, y)
{
  x = x[!is.na(x)]
  y = y[!is.na(y)]
  z = c(x, y)
  
  nx = as.numeric(length(x))
  ny = as.numeric(length(y))
  
  rx = rank(x, ties.method = 'average')
  ry = rank(y, ties.method = 'average')
  rall = rank(z, ties.method = 'average')
  
  stat = sum((rx-rall[1:nx])^2)/ny/(nx+ny) + sum((ry-rall[-c(1:nx)])^2)/nx/(nx+ny) - (4*nx*ny-1)/6/(nx+ny) 
  return(stat)
}

npardist2 = function(func, datavec, obid)
{
  processedvec = inputprocess(datavec)
  Out = NULL
  for(i in 1:ncol(processedvec))
  {
    Out = c(Out, rep(processedvec[1,i], processedvec[2,i]))
  }
  return(func(obid^2, Out^2))
}

mtsumtool = function(mt)
{
  cormt = cov(mt)
  diagcov = diag(cormt)
  for(i in 2:nrow(cormt))
  {
    for(j in 1:(i-1))
    {
      if(cormt[i,j]!=0)
      cormt[i,j]=cormt[i,j]/sqrt(diagcov[i]*diagcov[j])
    }
  }
  vct = cormt[lower.tri(cormt)]
  return(vct)
}

NetFeature = function(NetObj)
{
  Out = sapply(NetObj, mtsumtool)
  return(t(Out))
}

labelExType = function(dirnm)
{
  dirlist = list.dirs(path = paste('./', dirnm, sep = ''))[-1]
  dirlist = Cpcheck(dirlist)
  labelpre1 = sapply(dirlist, function(x)unlist(strsplit(x,'[/]'))[3])
  labelpre2 = sapply(labelpre1, function(x)unlist(strsplit(x,'[-]'))[3])
  labelout = t(sapply(labelpre2, function(x)unlist(strsplit(x,'[,]'))))
  colnames(labelout) = c('L1','L2','L3','L4')
  labelout = data.frame(protein_name = labelpre1,labelout)
  rownames(labelout) = c()
  return(labelout)
}

addlabel = function(dmat, clabel, cid)
{
  l = clabel[,cid]
  dout = data.frame(V1=l,dmat)
  return(dout)
}

c.svm = function(dat)
{
  c = 1:10
  out = sapply(c, function(x)svm.cv.10fold(dat, x))
  return(out)
}

PCkeep = function(pcr, th=0.9)
{
  rt = cumsum(pcr$sdev^2)/sum(pcr$sdev^2)
  Out = min(which(rt>th))
  return(Out)
}

Cpcheck = function(dirlist)
{
  dl = length(dirlist)
  dlogi = rep(TRUE, dl)
  for(i in 1:dl)
  {
    if(!(length(list.files(dirlist[i])) %in% c(72, 73, 74))) dlogi[i] = FALSE
  }
  if(sum(dlogi) < dl)
  {
    write.table(dirlist[!dlogi], file = paste('../', id_file, '_error.txt',sep = ''))
    return(dirlist[dlogi])
  }else{
    return(dirlist)
  }
}