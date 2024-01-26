##### The visualization of SHAP values.
options(stringsAsFactors = F)
library(reshape2)
library(ggplot2)
library(ggpubr)


melanet_spg <- readRDS("./data/spinglass/melanet_spg.RData")
samplePartition <- readRDS("./data/sample_partition.RData")

shap_statistic <- matrix(data = NA, nrow = 42, ncol = 3, dimnames = list(1:42, c("Cluster1", "Cluster2", "Cluster3")))

for(i in 1:length(melanet_spg)){
  shapfeat_htseq <- read.csv(paste("./data/python_related/shap/cmt", i, "shapfeat_htseq.csv", sep = ""), check.names = F, header = T)
  shapfeat_methy <- read.csv(paste("./data/python_related/shap/cmt", i, "shapfeat_methy.csv", sep = ""), check.names = F, header = T)
  tmp1 <- mean(abs(as.vector(as.matrix(shapfeat_htseq[which(samplePartition$cluster == 1),]))))
  tmp2 <- mean(abs(as.vector(as.matrix(shapfeat_htseq[which(samplePartition$cluster == 2),]))))
  tmp3 <- mean(abs(as.vector(as.matrix(shapfeat_htseq[which(samplePartition$cluster == 3),]))))
  tmp4 <- mean(abs(as.vector(as.matrix(shapfeat_methy[which(samplePartition$cluster == 1),]))))
  tmp5 <- mean(abs(as.vector(as.matrix(shapfeat_methy[which(samplePartition$cluster == 2),]))))
  tmp6 <- mean(abs(as.vector(as.matrix(shapfeat_methy[which(samplePartition$cluster == 3),]))))
  shap_statistic[2*i-1,] <- c(tmp1, tmp2, tmp3)
  shap_statistic[2*i,] <- c(tmp4, tmp5, tmp6)
}

data1 <- as.data.frame(shap_statistic[seq(1,42,2),])
data2 <- as.data.frame(shap_statistic[seq(2,42,2),])
rownames(data1) <- paste("Cmt", 1:21, sep = "")
rownames(data2) <- paste("Cmt", 1:21, sep = "")
data1$cmt <- rownames(data1)
data2$cmt <- rownames(data2)
data1 <- melt(data1, id.vars = c("cmt"))
data2 <- melt(data2, id.vars = c("cmt"))
data1$cmt <- factor(data1$cmt, levels = paste("Cmt", 1:21, sep = ""))
data2$cmt <- factor(data2$cmt, levels = paste("Cmt", 1:21, sep = ""))

pdf("./figure/shap_fig/shap_barplot.pdf", width = 21, height = 29.7)
p1 <- ggplot(data1, mapping = aes(x = cmt, y = value, fill = variable)) +
  geom_bar(stat="identity", position=position_dodge(0.75), width = 0.8) +
  labs(x='Community', y = 'Averaged abs SHAP values of all features', title = 'Gene expression profile') +
  theme_classic(base_size = 24) + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), legend.position = c(0.85,0.85)) +
  scale_fill_manual(values = c("Cluster1" = "#33A02C", "Cluster2" = "#1F78B4","Cluster3" = "#E31A1C")) +
  guides(fill=guide_legend(title=NULL))

p2 <- ggplot(data2, mapping = aes(x = cmt, y = value, fill = variable)) +
  geom_bar(stat="identity", position=position_dodge(0.75), width = 0.8) +
  labs(x='Community', y = 'Averaged abs SHAP values of all features', title = 'Methylation profile') + 
  theme_classic(base_size = 24) + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), legend.position = c(0.85,0.85)) +
  scale_fill_manual(values = c("Cluster1" = "#33A02C", "Cluster2" = "#1F78B4","Cluster3" = "#E31A1C")) +
  guides(fill=guide_legend(title=NULL))
ggarrange(p1, p2, nrow = 2, ncol = 1)
dev.off()

core_map_nodes <- read.csv("./data/melanoma_network/core_network_nodes.csv", header = T, check.names = F)
core_map_edges <- read.csv("./data/melanoma_network/core_network_edges.csv", header = T, check.names = F)

### Genes in the melanoma core network
curated_genes <- core_map_nodes$hgnc_symbol
curated_genes <- curated_genes[which(curated_genes != "")]
curated_genes <- curated_genes[which(!duplicated(curated_genes))] 


pdf("./figure/shap_fig/shap_values_top20.pdf", width = 21/2, height = 21/4)
lapply(1:length(melanet_spg), function(i){
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
  
  shapFhtseq <- within(shapFhtseq, {V1 <- factor(V1, levels = shapFhtseq$V1)})
  shapFmethy <- within(shapFmethy, {V1 <- factor(V1, levels = shapFmethy$V1)})
  
  ifcurated_exp <- ifelse(as.character(shapFhtseq$V1) %in% curated_genes, "*", "")
  ifcurated_mty <- ifelse(as.character(shapFmethy$V1) %in% curated_genes, "*", "")
  
  p1 <- ggplot(shapFhtseq, aes(x = V1, y = V2)) + 
    geom_bar(stat = "identity", fill = "#FFC7C7") + 
    labs(x = "Gene", y = "SHAP value", title = paste("Gene expression for cmt", i, sep = "")) + 
    theme(text = element_text(size = 10), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) +
    geom_text(aes(label = ifcurated_exp), position = position_dodge(0.9), colour = "red", size = 6)
  
  p2 <- ggplot(shapFmethy, aes(x = V1, y = V2)) + 
    geom_bar(stat = "identity", fill = "#68B0AB") + 
    labs(x = "Gene", y = "SHAP value", title = paste("Methylation for cmt", i, sep = "")) +
    theme(text = element_text(size = 10), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) +
    geom_text(aes(label = ifcurated_mty), position = position_dodge(0.9), colour = "red", size = 6)
  
  figure <- ggarrange(p1, p2, ncol=2, nrow=1)
  figure
})
dev.off()

shap_cmt <- matrix(data = NA, nrow = 21, ncol = 2, dimnames = list(paste("Cmt", 1:21, sep = ""), c("Gene expression", "Methylation")))

for(i in 1:length(melanet_spg)){
  shapFhtseq <- read.csv(paste("./data/python_related/shap/cmt", i, "reshtseq.csv", sep = ""), check.names = F, header = F)
  shapFmethy <- read.csv(paste("./data/python_related/shap/cmt", i, "resmethy.csv", sep = ""), check.names = F, header = F)
  shap_cmt[i,1] <- mean(abs(shapFhtseq$V2))
  shap_cmt[i,2] <- mean(abs(shapFmethy$V2))
}

shap_cmt <- as.data.frame(shap_cmt)
shap_cmt$cmt <- row.names(shap_cmt)
shap_cmt$cmt <- factor(shap_cmt$cmt, levels = paste("Cmt", 1:21, sep = ""))
shap_cmt <- melt(shap_cmt, id.vars = c("cmt"))

pdf("./figure/shap_fig/shap_cmt.pdf", width = 21, height = 29.7/2)
ggplot(shap_cmt, mapping = aes(x = cmt, y = value, fill = variable)) +
  geom_bar(stat="identity", position=position_dodge(0.75), width = 0.8) +
  labs(x='Community', y = 'Averaged abs SHAP values of all features') +
  theme_classic(base_size = 24) + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1), legend.position = c(0.85,0.85)) +
  scale_fill_manual(values = c("Gene expression" = "#FFC7C7", "Methylation" = "#68B0AB")) +
  guides(fill=guide_legend(title=NULL))
dev.off()