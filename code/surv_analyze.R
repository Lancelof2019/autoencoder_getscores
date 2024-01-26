##### Survival analysis
options(stringsAsFactors = F)
library(TCGAbiolinks)

clinicalInfo <- readRDS("./data/tcga_data_processed/clinical_info.RData")
samplePartition <- readRDS("./data/sample_partition.RData")

survivalInfo <- clinicalInfo[,c("shortLetterCode", "tumor_stage","vital_status","days_to_death","days_to_last_follow_up")]
survivalInfo$hc <- samplePartition$cluster # 病人层次聚类结果

survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage 0")] <- 0
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage i")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ia")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ib")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ic")] <- 1
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage ii")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iia")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iib")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iic")] <- 2
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iii")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iiia")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iiib")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iiic")] <- 3
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "stage iv")] <- 4
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "not reported")] <- "NA"
survivalInfo$tumor_stage[which(survivalInfo$tumor_stage == "i/ii nos")] <- "NA"
survivalInfo$tumor_stage[which(is.na(survivalInfo$tumor_stage))] <- "NA"

TCGAanalyze_survival(survivalInfo, clusterCol = "hc", color = c("#33A02C","#1F78B4","#E31A1C"), filename = "./figure/surv_analysis/survival_analysis.pdf", conf.int = F, width = 7, height = 7)

TCGAanalyze_survival(survivalInfo, clusterCol = "shortLetterCode", filename = "./figure/surv_analysis/survival_analysis_tumorType.pdf", conf.int = F, width = 7, height = 7)

TCGAanalyze_survival(survivalInfo, clusterCol = "tumor_stage", filename = "./figure/surv_analysis/survival_analysis_tumorStage.pdf", conf.int = F, width = 7, height = 7)
