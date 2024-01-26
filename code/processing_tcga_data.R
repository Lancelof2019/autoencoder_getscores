##### TCGA data processing
options(stringsAsFactors = F)
library(TCGAbiolinks)
library(SummarizedExperiment)
library(limma)
library(biomaRt)

#### Collect gene expression data
query_exp <- GDCquery(project = "TCGA-SKCM", 
                      data.category = "Transcriptome Profiling",
                      data.type = "Gene Expression Quantification",
                      workflow.type = "HTSeq - FPKM")
GDCdownload(query_exp) 
data_exp <- GDCprepare(query_exp) 
exp_assay <- as.data.frame(assay(data_exp)) # Gene expression matrix
exp_colData <- as.data.frame(colData(data_exp)) # Patient annotation(472 patients)
exp_rowRanges <- as.data.frame(rowRanges(data_exp)) # Gene annotation

#### Collect DNA methylation data
query_mty <- GDCquery(project = "TCGA-SKCM",
                      data.category = "DNA Methylation",
                      platform = "Illumina Human Methylation 450")
GDCdownload(query_mty)
data_mty <- GDCprepare(query_mty)
mty_assay <- as.data.frame(assay(data_mty)) # DNA methylation matrix
mty_colData <- as.data.frame(colData(data_mty)) # Patient annotation(475 patients)
saveRDS(mty_colData, "./data/tcga_data/mty_colData.RData")
mty_rowRanges <- as.data.frame(rowRanges(data_mty)) # cg probe annotation

#### Collect gene mutation data
maf <- GDCquery_Maf("SKCM", pipelines = "muse") # Mutation annotation
saveRDS(maf, "./data/tcga_data/maf.RData")

#### Collect copy number variation data
query_cnv <- GDCquery(project = "TCGA-SKCM",
                      data.category = "Copy Number Variation",
                      data.type = "Gene Level Copy Number Scores")
GDCdownload(query_cnv)
data_cnv <- GDCprepare(query_cnv)
data_cnv <- as.data.frame(data_cnv) # Gene-level copy number values

#### Collect clinical data
clinical <- GDCquery_clinic(project = "TCGA-SKCM", type = "clinical")

#### Collect clinical radiation and drug therapy data
query <- GDCquery(project = "TCGA-SKCM", 
                  data.category = "Clinical",
                  data.type = "Clinical Supplement",
                  data.format = "BCR Biotab"
)
GDCdownload(query)
sckm.tab.all <- GDCprepare(query)

therapy <- sckm.tab.all$clinical_drug_skcm # clinical drug therapy
saveRDS(therapy, "./data/tcga_data/therapy.RData")
therapy$pharmaceutical_therapy_type # Therapy types

radiation <- sckm.tab.all$clinical_radiation_skcm # Clinical radiation therapy
saveRDS(radiation, "./data/tcga_data/radiation.RData")

#### Process gene expression data
colnames(exp_assay) <- substr(colnames(exp_assay), 1, 16)
exp_assay$SYMBOL <- exp_rowRanges[row.names(exp_assay), "external_gene_name"] 
exp_assay <- exp_assay[,c(473,1:472)]
exp_assay <- as.matrix(exp_assay)
row.names(exp_assay) <- exp_assay[,1]
exp_assay <- exp_assay[,2:473] # Convert Ensemble ID to corresponding gene symbols
exp_assay <- matrix(as.numeric(exp_assay), nrow = nrow(exp_assay), dimnames = list(row.names(exp_assay), colnames(exp_assay)))
exp_assay <- avereps(exp_assay) # If the gene corresponds to multiple gene expression values, take the average value

#### Process DNA methylation data
data_mty <- subset(data_mty, subset = (rowSums(is.na(assay(data_mty)))==0)) 
data_mty <- subset(data_mty, subset = (as.data.frame(rowRanges(data_mty))$Gene_Symbol!="."))
mty_assay <- as.data.frame(assay(data_mty))
mty_rowRanges <- as.data.frame(rowRanges(data_mty))
mty_symbol <- strsplit(mty_rowRanges$Gene_Symbol, split = ";")
names(mty_symbol) <- row.names(mty_rowRanges)
mty_symbol <- lapply(mty_symbol, FUN = function(x){x<-unique(x)})
mty_symbol <- as.matrix(unlist(mty_symbol))
row.names(mty_symbol) <- substr(row.names(mty_symbol), 1, 10)
mty_symbol <- data.frame("probe" = row.names(mty_symbol),"SYMBOL" = mty_symbol[,1])
mty_assay$probe <- row.names(mty_assay)
mty_assay <- merge(mty_assay, mty_symbol, by.x = "probe", by.y = "probe") 
mty_mat <- as.matrix(mty_assay[,2:476])
colnames(mty_mat) <- substr(colnames(mty_mat), 1, 16)
row.names(mty_mat) <- mty_assay$SYMBOL # Convert probe ID to corresponding gene symbols
mty_mat <- matrix(as.numeric(mty_mat), nrow = nrow(mty_mat), dimnames = list(row.names(mty_mat), colnames(mty_mat)))
mty_mat <- avereps(mty_mat) # If the gene corresponds to multiple methylation values, take the average value

#### Process gene mutation data
maf <- maf[,c("Tumor_Sample_Barcode","Hugo_Symbol","Gene","Variant_Classification")]
rnames <- unique(maf$Hugo_Symbol)
cnames <- unique(maf$Tumor_Sample_Barcode)
snv_count <- matrix(data = 0, nrow = length(rnames), ncol = length(cnames), dimnames = list(rnames,cnames)) 
# Calculate the frequency of genes' variants
for(i in 1:nrow(maf)){
  rname <- maf[i,]$Hugo_Symbol
  cname <- maf[i,]$Tumor_Sample_Barcode
  snv_count[rname, cname] <- snv_count[rname,cname] + 1
}
colnames(snv_count) <- substr(colnames(snv_count), 1, 16)

#### Process copy number variation data
row.names(data_cnv) <- substr(data_cnv[,1], 1, 15)
data_cnv <- data_cnv[,4:ncol(data_cnv)]
colnames(data_cnv) <- substr(colnames(data_cnv), 1, 16)
# Convert Ensemble ID to corresponding gene symbols
# Using biomaRt for gene ID conversion will cause some loss
ensembl <- useMart("ensembl",dataset="hsapiens_gene_ensembl")
cnv_df <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
                filters = c("ensembl_gene_id"),
                values = row.names(data_cnv),
                mart = ensembl)
cnv_df <- cnv_df[which(cnv_df$hgnc_symbol !=''),]
data_cnv$emsembl <- row.names(data_cnv)
data_cnv <- merge(x = data_cnv, y = cnv_df, by.x = "emsembl", by.y = "ensembl_gene_id")
data_cnv <- as.matrix(data_cnv)
row.names(data_cnv) <- data_cnv[,474]
data_cnv <- data_cnv[,2:473]  # Convert Ensemble ID to corresponding gene symbols
data_cnv <- matrix(as.numeric(data_cnv), nrow = nrow(data_cnv), dimnames = list(row.names(data_cnv), colnames(data_cnv)))
data_cnv <- data_cnv[!duplicated(row.names(data_cnv)),] # Only one gene PRAMEF7 has duplicate copy number variation value, and the duplicate value is the same

#### Keep patient samples with four genomic profiles
samples <- intersect(colnames(exp_assay), colnames(mty_mat)) 
samples <- intersect(samples, colnames(snv_count))
samples <- intersect(samples, colnames(data_cnv))
saveRDS(samples, "./data/tcga_data_processed/samples.RData")

#### Integrate the genomic data into the network
exp_assay <- exp_assay[,samples]
mty_mat <- mty_mat[,samples]
snv_count <- snv_count[,samples]
data_cnv <- data_cnv[,samples]
genes <- as.character(unlist(vertex.attributes(graph_comp))) # Nodes in the maximum connected subgraph

y1 <- which(row.names(exp_assay) %in% genes)
exp_intgr <- exp_assay[y1,] 
exp_intgr <- t(exp_intgr) # Gene expression data matrix
exp_intgr <- log10(exp_intgr + 1) # 1og10 conversion
saveRDS(exp_intgr, "./data/tcga_data_processed/exp_intgr.RData")

y2 <- which(row.names(mty_mat) %in% genes)
mty_intgr <- mty_mat[y2,]
mty_intgr <- t(mty_intgr) # DNA methylation data matrix
saveRDS(mty_intgr, "./data/tcga_data_processed/mty_intgr.RData")

y3 <- which(row.names(snv_count) %in% genes)
snv_intgr <- snv_count[y3,]
snv_intgr <- t(snv_intgr) # Gene mutation data matrix
saveRDS(snv_intgr, "./data/tcga_data_processed/snv_intgr.RData")

y4 <- which(row.names(data_cnv) %in% genes)
cnv_intgr <- data_cnv[y4,]
cnv_intgr <- t(cnv_intgr) # Copy number variation data matrix
saveRDS(cnv_intgr, "./data/tcga_data_processed/cnv_intgr.RData")

clinicalInfo <- mty_colData
row.names(clinicalInfo) <- clinicalInfo$sample
clinicalInfo <- clinicalInfo[samples,]
saveRDS(clinicalInfo, "./data/tcga_data_processed/clinical_info.RData")













