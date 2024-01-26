### The frequency of simple nucleotide variation (SNV) and copy number variation (CNV) per patient cluster.
options(stringsAsFactors = F)
library(ggplot2)
library(ggpubr)

snv_intgr <- readRDS("./data/tcga_data_processed/snv_intgr.RData")
cnv_intgr <- readRDS("./data/tcga_data_processed/cnv_intgr.RData")
samplePartition <- readRDS("./data/sample_partition.RData")

snv_count <- apply(snv_intgr, 1, sum)
snv_count <- as.data.frame(snv_count)
snv_count$cluster <- as.factor(samplePartition$cluster)

cnv_gain_loss_count <- apply(cnv_intgr, 1, function(x){length(which(x!=0))})
cnv_gain_loss_count <- as.data.frame(cnv_gain_loss_count)
cnv_gain_loss_count$cluster <- as.factor(samplePartition$cluster)

# cnv_count <- matrix(nrow = length(colOrder), ncol = 2, dimnames = list(1:length(colOrder), c("gain","loss")))
# 
# for(i in 1:nrow(cnv_count)){
#   mat <- cnv_intgr[samplePartition$cluster == i,]
#   t1 <- length(which(mat == 1)) / (nrow(mat)*ncol(mat)) * 100
#   t2 <- length(which(mat == -1)) / (nrow(mat)*ncol(mat)) * 100
#   cnv_count[i,] <- c(t1,t2)
# }
# cnv_count <- as.data.frame(cnv_count)
# cnv_count$cluster <- row.names(cnv_count)
# 
# cnv_count<-melt(cnv_count,
#                 id.vars = c("cluster"),
#                 measure.vars = c("gain","loss"),
#                 variable.name = "state")

p1 <- ggplot(snv_count, aes(x = cluster, y = snv_count)) + geom_violin(outlier.shape = NA) + geom_boxplot(outlier.shape = NA, width=0.2) + scale_y_continuous(limits = quantile(snv_count$snv_count, c(0.1, 0.9))) + labs(x="Cluster", y = "Snv frequency", title = "Snv frequency per cluster") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black'))

p2 <- ggplot(cnv_gain_loss_count, aes(x = cluster, y = cnv_gain_loss_count)) + geom_violin(outlier.shape = NA) + geom_boxplot(outlier.shape = NA, width=0.2) + scale_y_continuous(limits = quantile(cnv_gain_loss_count$cnv_gain_loss_count, c(0.1, 0.9))) + labs(x="Cluster", y = "Cnv frequency", title = "Cnv frequency per cluster") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black'))

# p2 <- ggplot(cnv_gain_loss_count,aes(x = cluster,y = value,fill = state)) + geom_bar(stat='identity', position='stack') + labs(x="Cluster", y = "Cnv proportion (%)", title = "Cnv proportion per cluster") + theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black'), legend.position = c(0.9,0.9))

pdf("./figure/snv_cnv.pdf", width = 21/2 , height = 21/4)
figure <- ggarrange(p1, p2, ncol=2, nrow=1)
figure
dev.off()