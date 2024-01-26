##### Patient stratification
options(stringsAsFactors = F)
library(NbClust)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)
library(tidyverse)
library(maftools)

samples <- readRDS("./data/tcga_data_processed/samples.RData")
cmtScores <- read.csv("./data/python_related/result/communityScores.csv", check.names = F, header = F)
exp_intgr <- readRDS("./data/tcga_data_processed/exp_intgr.RData")
mty_intgr <- readRDS("./data/tcga_data_processed/mty_intgr.RData")
snv_intgr <- readRDS("./data/tcga_data_processed/snv_intgr.RData")
cnv_intgr <- readRDS("./data/tcga_data_processed/cnv_intgr.RData")
clinicalInfo <- readRDS("./data/tcga_data_processed/clinical_info.RData")
therapy <- readRDS("./data/tcga_data/therapy.RData")
radiation <- readRDS("./data/tcga_data/radiation.RData")
melanet_cmt <- readRDS("./data/spinglass/melanet_cmt.RData")

row.names(cmtScores) <- samples
colnames(cmtScores) <- paste0("cmt", 1:ncol(cmtScores))
saveRDS(cmtScores, "./data/community_scores.RData")

### Determine the best number of clusters
nc <- NbClust(scale(cmtScores), distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete", index = "all")

pdf("./figure/best_number_of_clusters.pdf", width = 7, height = 7)
ggplot(data.frame(cluster = factor(nc$Best.nc[1,])), aes(x = cluster)) + geom_bar(stat = "count", fill = "#C1BFBF") + labs(x = "Number of clusters", y = "Number of criteria", title = "Number of clusters chosen by 26 criteria") + theme(text = element_text(size = 18), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + 
  scale_y_continuous(breaks = seq(0,14,2), limits = c(0,14))
dev.off()

### Community scores of clustered patients
tumorType <- clinicalInfo[,"shortLetterCode"]
tumorStage <- clinicalInfo[,"tumor_stage"]
tumorStage[-grep("^stage", tumorStage)] <- NA
tumorStage <- gsub("^stage ", "", tumorStage)
tumorStage <- gsub("[a-c]$", "", tumorStage)

tumorStage <- clinicalInfo[,"tumor_stage"]
tumorStage[-grep("^stage", tumorStage)] <- NA
tumorStage <- gsub("^stage ", "", tumorStage)
tumorStage <- gsub("[a-c]$", "", tumorStage)

therapy <- therapy[3:nrow(therapy),]
ifTherapy <- substr(samples, 1, 12) %in% therapy$bcr_patient_barcode
ifTherapy <- ifelse(ifTherapy, "Yes", "No")

radiation <- radiation[3:nrow(radiation),]
ifRadiation <- substr(samples, 1, 12) %in% radiation$bcr_patient_barcode
ifRadiation <- ifelse(ifRadiation, "Yes", "No")

therapy_type <- sapply(substr(samples, 1, 12), function(x){paste(sort(unique(therapy$pharmaceutical_therapy_type[therapy$bcr_patient_barcode == x])),collapse = ";")})

tr_df <- data.frame(sample = samples, theray = ifTherapy, radiation = ifRadiation, therapy_type = therapy_type)
saveRDS(tr_df, "./data/therapy_radiation_df.RData")

tumorType_col_fun <- c("TM" = "#CC79A7", "TP" = "#0072B2")
tumorStage_col_fun <- c("0" = "#FAFCC2", "i" = "#FFEFA0", "ii" = "#FFD57E", "iii" = "#FCA652", "iv" = "#AC4B1C")
ifTherapy_col_fun <- c("Yes" = "red", "No" = "gray")
ifRadiation_col_fun <- c("Yes" = "red", "No" = "gray")

topAnno <- HeatmapAnnotation(Therapy = ifTherapy, Radiation = ifRadiation, `Tumor type` = tumorType, `Tumor stage` = tumorStage, col = list(Therapy = ifTherapy_col_fun, Radiation = ifRadiation_col_fun, `Tumor type` = tumorType_col_fun, `Tumor stage` = tumorStage_col_fun), border = T, show_annotation_name = T)
ht = Heatmap(t(scale(cmtScores)), 
             name = "Community score", 
             show_column_names = F,
             # top_annotation = topAnno,
             clustering_distance_columns = "euclidean",
             clustering_method_columns = "complete",
             column_split = 3,
             column_title = "%s",
)

pdf("./figure/heatmap_cmtScores.pdf", width = 7, height = 7)
draw(ht, merge_legends = TRUE)
dev.off()


ht = draw(ht)
rowOrder <- row_order(ht)
colOrder <- column_order(ht)
samplePartition <- data.frame(cluster = rep(1:length(colOrder), lengths(colOrder)), sampleID = unlist(colOrder))
samplePartition <- samplePartition[order(samplePartition$sampleID),]
saveRDS(samplePartition, "./data/sample_partition.RData")

tcgaData <- list(exp = t(exp_intgr),
                 mty = t(mty_intgr),
                 snv = t(snv_intgr),
                 cnv = t(cnv_intgr))
tumorType1 <- tumorType[unlist(colOrder)]
tumorStage1 <- tumorStage[unlist(colOrder)]
ifTherapy1 <- ifTherapy[unlist(colOrder)]
ifRadiation1 <- ifRadiation[unlist(colOrder)]
topAnno1 <- HeatmapAnnotation(Therapy = ifTherapy1, Radiation = ifRadiation1, `Tumor type` = tumorType1, `Tumor stage` = tumorStage1, col = list(Therapy = ifTherapy_col_fun, Radiation = ifRadiation_col_fun, `Tumor type` = tumorType_col_fun, `Tumor stage` = tumorStage_col_fun), border = T, show_annotation_name = T)
for(i in 1:length(tcgaData)){
  dmat <- tcgaData[[i]]
  dmat <- dmat[,unlist(colOrder)]
  if(i == 1){
    col_fun <- colorRamp2(c(min(dmat), max(dmat)),c("#FFFFFF","#FFC7C7"))
  }
  else if(i == 2){
    col_fun <- colorRamp2(c(min(dmat), max(dmat)),c("#D9EBEA","#68B0AB"))
  }
  else if(i == 3){
    col_fun <- colorRamp2(c(min(dmat), mean(dmat), max(dmat)),c("#DBDBDB","#c2a5cf","#7b3294"))         
  }
  else{
    col_fun <- colorRamp2(c(min(dmat), mean(dmat), max(dmat)),c("blue","white","red"))
  }
  rht <- Heatmap(dmat,
                 name = names(tcgaData)[i],
                 col = col_fun,
                 top_annotation = topAnno1,
                 show_column_names = F,
                 show_row_names = F,
                 cluster_columns = F,
                 cluster_rows = T,
                 column_split = rep(1:length(colOrder), lengths(colOrder)))
  pdf(paste("./figure/heatmap_", names(tcgaData)[i], ".pdf", sep = ""));
  draw(rht, merge_legends = TRUE);
  dev.off()
}

plot_genes_exp <- c() 
plot_genes_mty <- c()
ht_list <- list()
col_fun <- list()
for(i in 1:21){
  shapFhtseq <- read.csv(paste("./data/python_related/shap/cmt", i, "reshtseq.csv", sep = ""), check.names = F, header = F) # 关于基因表达的每个社区的SHAP得分
  shapFmethy <- read.csv(paste("./data/python_related/shap/cmt", i, "resmethy.csv", sep = ""), check.names = F, header = F) # 关于DNA甲基化的每个社区的SHAP得分
  
  shapFhtseq <- shapFhtseq[order(shapFhtseq[,2],decreasing = T),]
  shapFmethy <- shapFmethy[order(shapFmethy[,2],decreasing = T),]
  
  if(nrow(shapFhtseq) >= 20){
    shapFhtseq <- shapFhtseq[1:20,]
  }
  if(nrow(shapFmethy) >= 20){
    shapFmethy <- shapFmethy[1:20,]
  }
  
  shapFhtseq[,1] <- gsub('["]', '', shapFhtseq[,1])
  shapFmethy[,1] <- gsub('["]', '', shapFmethy[,1])
  
  plot_genes_exp <- union(plot_genes_exp, shapFhtseq[,1])
  plot_genes_mty <- union(plot_genes_mty, shapFmethy[,1])
}
plot_genes_union <- union(plot_genes_exp, plot_genes_mty)

for(i in 1:length(tcgaData)){
  plot_genes <- c()
  if(i == 1){
    plot_genes <- plot_genes_exp
  }
  else if(i == 2){
    plot_genes <- plot_genes_mty
  }
  else if(i == 3){
    plot_genes <- plot_genes_union
  }
  else{
    plot_genes <- plot_genes_union
  }
  dmat <- tcgaData[[i]]
  plot_genes <- plot_genes[which(plot_genes %in% row.names(dmat))]
  dmat <- dmat[plot_genes,]
  if(i == 1){
    col_fun <- colorRamp2(c(min(dmat), max(dmat)),c("#FFFFFF","#FFC7C7"))
  }
  else if(i == 2){
    col_fun <- colorRamp2(c(min(dmat), max(dmat)),c("#D9EBEA","#68B0AB"))
  }
  else if(i == 3){
    col_fun <- colorRamp2(c(min(dmat), mean(dmat), max(dmat)),c("#DBDBDB","#c2a5cf","#7b3294"))         
  }
  else{
    col_fun <- colorRamp2(c(min(dmat), mean(dmat), max(dmat)),c("blue","white","red"))
  }
  if(i==1){
    ht_list[[i]] <- Heatmap(dmat, 
                            name = names(tcgaData)[i],
                            col = col_fun,
                            top_annotation = topAnno1,
                            show_column_names = F,
                            show_row_names = F,
                            cluster_columns = F,
                            column_split = rep(1:length(colOrder),lengths(colOrder)))
  }
  else{
    ht_list[[i]] <- Heatmap(dmat, 
                            name = names(tcgaData)[i],
                            col = col_fun,
                            # top_annotation = topAnno1,
                            show_column_names = F,
                            show_row_names = F,
                            cluster_columns = F,
                            column_split = rep(1:length(colOrder),lengths(colOrder)))
  }
}
pdf(paste("./figure/heatmap_union.pdf", sep = ""));
draw(ht_list[[1]] %v% ht_list[[2]] %v% ht_list[[3]] %v% ht_list[[4]], merge_legends = TRUE);
dev.off()

### The mutation profiles of the top-ranking features in the network communities
maf <- readRDS("./data/tcga_data/maf.RData")
maf <- maf %>% read.maf
annotationDat <- data.frame(samples = samples, Cluster = samplePartition$cluster)
tumor_samples_barcode <- data.frame(Tumor_Sample_Barcode = maf@clinical.data[["Tumor_Sample_Barcode"]])
tumor_samples_barcode$samples <- substr(tumor_samples_barcode$Tumor_Sample_Barcode, 1, 16)
annotationDat <- merge(annotationDat, tumor_samples_barcode, by.x = "samples", by.y = "samples", all.y = T)
annotationDat$Cluster[which(is.na(annotationDat$Cluster))] <- "NA"

cluster_col <- c("#33A02C", "#1F78B4", "#E31A1C", "gray")
names(cluster_col) <- c("1", "2", "3", "NA")

for(i in 1:length(melanet_cmt)){
  shapFhtseq <- read.csv(paste("./data/python_related/shap/cmt", i, "reshtseq.csv", sep = ""), check.names = F, header = F) # 关于基因表达的每个社区的SHAP得分
  shapFmethy <- read.csv(paste("./data/python_related/shap/cmt", i, "resmethy.csv", sep = ""), check.names = F, header = F) # 关于DNA甲基化的每个社区的SHAP得分
  
  shapFhtseq <- shapFhtseq[order(shapFhtseq[,2],decreasing = T),]
  shapFmethy <- shapFmethy[order(shapFmethy[,2],decreasing = T),]
  
  if(nrow(shapFhtseq) >= 20){
    shapFhtseq <- shapFhtseq[1:20,]
  }
  if(nrow(shapFmethy) >= 20){
    shapFmethy <- shapFmethy[1:20,]
  }
  
  shapFhtseq[,1] <- gsub('["]', '', shapFhtseq[,1])
  shapFmethy[,1] <- gsub('["]', '', shapFmethy[,1])
  
  genes <- union(shapFhtseq[,1], shapFmethy[,1])
  pdf(paste("./figure/maf_oncoplot/cmt", i,"_maf_oncoplot.pdf", sep = ""))
  oncoplot(maf = maf, top = 20, genes = genes, removeNonMutated = T, annotationDat = annotationDat, clinicalFeatures = "Cluster", annotationColor = list(Cluster = cluster_col), sortByAnnotation = T, gene_mar = 6, fontSize = 0.7)
  dev.off()
}

sig_genes <- c("BRAF", "MAP2K1", "MAP2K2", "MAP2K3", "MAP2K4", "MAP2K5", "MAP2K7", "NRAS", "CDKN2A", "TP53", "NF1", "PTEN", "KIT ")
pdf("./figure/maf_oncoplot/maf_oncoplot_sig_genes.pdf")
oncoplot(maf = maf, genes = sig_genes, removeNonMutated = T, annotationDat = annotationDat, clinicalFeatures = "Cluster", annotationColor = list(Cluster = cluster_col), sortByAnnotation = T, gene_mar = 6, fontSize = 0.7)
dev.off()