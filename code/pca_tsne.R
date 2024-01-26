### pca tsne visualization
options(stringsAsFactors = F)
library(ggplot2)
library(Rtsne)
library(ggpubr)

cmtScores <- readRDS("./data/community_scores.RData")
exp_intgr <- readRDS("./data/tcga_data_processed/exp_intgr.RData")
mty_intgr <- readRDS("./data/tcga_data_processed/mty_intgr.RData")
clinicalInfo <- readRDS("./data/tcga_data_processed/clinical_info.RData")
samplePartition <- readRDS("./data/sample_partition.RData")

cmtScores_pca <- prcomp(cmtScores, scale. = F, retx = T)
exp_pca <- prcomp(exp_intgr, scale. = F, retx = T)
mty_pca<-prcomp(mty_intgr, scale. = F, retx = T)

tumorType <- clinicalInfo[,"shortLetterCode"]
cmtScores_pca_df <- data.frame(cmtScores_pca$x, cluster = as.factor(samplePartition$cluster), tumor_type = as.factor(tumorType))
exp_pca_df <- data.frame(exp_pca$x, cluster = as.factor(samplePartition$cluster), tumor_type = as.factor(tumorType))
mty_pca_df <- data.frame(mty_pca$x, cluster = as.factor(samplePartition$cluster), tumor_type = as.factor(tumorType))


p1 <- ggplot(cmtScores_pca_df,aes(x = PC1, y = PC2, col = cluster, shape = tumor_type)) + geom_point() + labs(x = "PC1", y = "PC2", title = "Community score pca") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + scale_color_manual(values = c("1"="#33A02C", "2"="#1F78B4", "3"="#E31A1C"))
p2 <- ggplot(exp_pca_df, aes(x = PC1, y = PC2, col = cluster, shape = tumor_type)) + geom_point() + labs(x = "PC1", y = "PC2",title = "Gene expression pca") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + scale_color_manual(values = c("1"="#33A02C", "2"="#1F78B4", "3"="#E31A1C"))
p3 <- ggplot(mty_pca_df, aes(x = PC1, y = PC2, col = cluster, shape = tumor_type)) + geom_point() + labs(X = "PC1", y = "PC2", title = "Methylation pca") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + scale_color_manual(values = c("1"="#33A02C", "2"="#1F78B4", "3"="#E31A1C"))

# t-SNE
cmtScores_tsne <- Rtsne(cmtScores,
                        dims = 2,
                        pca = F,
                        perplexity = 10,
                        theta = 0.0,
                        max_iter = 1000)
exp_tsne <- Rtsne(exp_intgr,
                  dims = 2,
                  pca = F,
                  perplexity = 10,
                  theta = 0.0,
                  max_iter = 1000)
mty_tsne <- Rtsne(mty_intgr,
                  dims = 2,
                  pca = F,
                  perplexity = 10,
                  theta = 0.0,
                  max_iter = 1000)

cmtScores_tsne_df <- data.frame(cmtScores_tsne$Y, cluster = as.factor(samplePartition$cluster), tumor_type = as.factor(tumorType))
exp_tsne_df <- data.frame(exp_tsne$Y, cluster = as.factor(samplePartition$cluster), tumor_type = as.factor(tumorType))
mty_tsne_df <- data.frame(mty_tsne$Y, cluster = as.factor(samplePartition$cluster), tumor_type = as.factor(tumorType))

p4 <- ggplot(cmtScores_tsne_df, aes(x = X1, y = X2, col = cluster, shape = tumor_type)) + geom_point() + labs(x = "T-SNE1", y = "T-SNE2", title = "Community score t-sne") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + scale_color_manual(values = c("1"="#33A02C", "2"="#1F78B4", "3"="#E31A1C"))

p5 <- ggplot(exp_tsne_df, aes(x = X1, y = X2, col = cluster, shape = tumor_type)) + geom_point() + labs(x = "T-SNE1", y = "T-SNE2", title = "Gene expression t-sne") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + scale_color_manual(values = c("1"="#33A02C", "2"="#1F78B4", "3"="#E31A1C"))

p6 <- ggplot(mty_tsne_df, aes(x = X1, y = X2, col = cluster, shape = tumor_type)) + geom_point() + labs(x = "T-SNE1", y = "T-SNE2", title = "Methylation t-sne") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + scale_color_manual(values = c("1"="#33A02C", "2"="#1F78B4", "3"="#E31A1C"))

pdf("./figure/pca_tsne.pdf",width = 20,height = 10)
figure <- ggarrange(p1, p2, p3, p4, p5, p6, ncol = 3,nrow = 2)
figure
dev.off()