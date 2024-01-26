options(stringsAsFactors = F)
library(vcd)

##### Statistical tests on clinical features
samples <- readRDS("./data/tcga_data_processed/samples.RData")
clinicalInfo <- readRDS("./data/tcga_data_processed/clinical_info.RData")
samplePartition <- readRDS("./data/sample_partition.RData")
snv_intgr <- readRDS("./data/tcga_data_processed/snv_intgr.RData")
cnv_intgr <- readRDS("./data/tcga_data_processed/cnv_intgr.RData")
therapy <- readRDS("./data/tcga_data/therapy.RData")
radiation <- readRDS("./data/tcga_data/radiation.RData")

test <- clinicalInfo[,c("shortLetterCode", "tumor_stage", "gender", "age_at_index")]

test$cluster <- samplePartition$cluster
test$snv_count <- apply(snv_intgr, 1, sum)
test$cnv_gain_loss_count<-apply(cnv_intgr, 1, function(x){length(which(x!=0))})

# test$tumor_stage[-grep("^stage", test$tumor_stage)] <- NA
# test$tumor_stage <- gsub("^stage ", "", test$tumor_stage)
# test$tumor_stage <- gsub("[a-c]$", "", test$tumor_stage)

test$tumor_stage[which(test$tumor_stage == "stage 0")] <- 0
test$tumor_stage[which(test$tumor_stage == "stage i")] <- 1
test$tumor_stage[which(test$tumor_stage == "stage ia")] <- 1
test$tumor_stage[which(test$tumor_stage == "stage ib")] <- 1
test$tumor_stage[which(test$tumor_stage == "stage ic")] <- 1
test$tumor_stage[which(test$tumor_stage == "stage ii")] <- 2
test$tumor_stage[which(test$tumor_stage == "stage iia")] <- 2
test$tumor_stage[which(test$tumor_stage == "stage iib")] <- 2
test$tumor_stage[which(test$tumor_stage == "stage iic")] <- 2
test$tumor_stage[which(test$tumor_stage == "stage iii")] <- 3
test$tumor_stage[which(test$tumor_stage == "stage iiia")] <- 3
test$tumor_stage[which(test$tumor_stage == "stage iiib")] <- 3
test$tumor_stage[which(test$tumor_stage == "stage iiic")] <- 3
test$tumor_stage[which(test$tumor_stage == "stage iv")] <- 4
test$tumor_stage[which(test$tumor_stage == "not reported")] <- NA
test$tumor_stage[which(test$tumor_stage == "i/ii nos")] <- NA

test$age <- NA
test$age[which(test$age_at_index < 20)] <- "<20"
test$age[which(test$age_at_index >= 20 & test$age_at_index <= 60)] <- "[20,60]"
test$age[which(test$age_at_index >60)] <- ">60"

therapy <- therapy[3:nrow(therapy),]
ifTherapy <- substr(samples, 1, 12) %in% therapy$bcr_patient_barcode
ifTherapy <- ifelse(ifTherapy, "Yes", "No")

radiation <- radiation[3:nrow(radiation),]
ifRadiation <- substr(samples, 1, 12) %in% radiation$bcr_patient_barcode
ifRadiation <- ifelse(ifRadiation, "Yes", "No")

test$ifTherapy <- ifTherapy
test$ifRadiation <- ifRadiation


kruskal.test(age_at_index ~ cluster, data = test)
kruskal.test(snv_count ~ cluster, data = test)
kruskal.test(cnv_gain_loss_count ~ cluster, data = test)

kruskal.test(age_at_index ~ cluster, data = test, subset = cluster %in% c(1,2))
kruskal.test(snv_count ~ cluster, data = test, subset = cluster %in% c(1,2))
kruskal.test(cnv_gain_loss_count ~ cluster, data = test, subset = cluster %in% c(1,2))

kruskal.test(age_at_index ~ cluster, data = test, subset = cluster %in% c(2,3))
kruskal.test(snv_count ~ cluster, data = test, subset = cluster %in% c(2,3))
kruskal.test(cnv_gain_loss_count ~ cluster, data = test, subset = cluster %in% c(2,3))

kruskal.test(age_at_index ~ cluster, data = test, subset = cluster %in% c(1,3))
kruskal.test(snv_count ~ cluster, data = test, subset = cluster %in% c(1,3))
kruskal.test(cnv_gain_loss_count ~ cluster, data = test, subset = cluster %in% c(1,3))

tumor_type_table <- xtabs( ~ cluster + shortLetterCode,data = test)
chisq.test(tumor_type_table)
assocstats(tumor_type_table)
tumor_type_table

tumor_stage_table <- xtabs( ~ cluster + tumor_stage,data = test)
chisq.test(tumor_stage_table)
assocstats(tumor_stage_table)
tumor_stage_table

gender_table <- xtabs( ~ cluster + gender,data = test)
chisq.test(gender_table)
assocstats(gender_table)
gender_table

age_table <- xtabs( ~ cluster + age, data = test)
chisq.test(age_table)
assocstats(age_table)
age_table

ifTherapy_table <- xtabs( ~ cluster + ifTherapy, data = test)
chisq.test(ifTherapy_table)
assocstats(ifTherapy_table)
ifTherapy_table

ifRadiation_table <- xtabs( ~ cluster + ifRadiation, data = test)
chisq.test(ifRadiation_table)
assocstats(ifRadiation_table)
ifRadiation_table
