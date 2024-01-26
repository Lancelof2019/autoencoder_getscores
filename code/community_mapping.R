options(stringsAsFactors = F)

exp_intgr <- readRDS("./data/tcga_data_processed/exp_intgr.RData")
mty_intgr <- readRDS("./data/tcga_data_processed/exp_intgr.RData")

hfeat <- colnames(exp_intgr) # The features of gene expression data
mfeat <- colnames(mty_intgr) # The features of DNA methylation data
selected_features <- matrix(ncol = 3, nrow=30000) # Matrix of 3 columns; column1: community, column2: genomic type, column3: mapped component
len <- 0
j <- 1
indx <- NULL
for(i in 1:length(melanet_cmt)){
  cmt = melanet_cmt[[i]]
  # Start mapping
  # Gene expression
  j = j + length(indx)
  indx = NULL
  indx = which(hfeat %in% cmt)
  if (length(indx) != 0){
    len = len +length(indx)
    selected_features[j:len,1] = i
    selected_features[j:len,2] = 1
    selected_features[j:len,3] = indx
  }
  
  # DNA methylation
  j = j+length(indx)
  indx = NULL
  indx = which(mfeat %in% cmt)
  if (length(indx) != 0){
    len = len + length(indx)
    selected_features[j:len,1] = i
    selected_features[j:len,2] = 2
    selected_features[j:len,3] = indx
  }
}
selected_features <- na.omit(selected_features)
write.csv(selected_features, "./data/python_data/selected_features.csv", row.names = F)
write.csv(colnames(exp_intgr), "./data/python_data/exp_feature_names.csv", row.names = F)
write.csv(colnames(mty_intgr), "./data/python_data/mty_feature_names.csv", row.names = F) 
write.csv(exp_intgr, "./data/python_data/exp_intgr.csv")
write.csv(mty_intgr, "./data/python_data/mty_intgr.csv")