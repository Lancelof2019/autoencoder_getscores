##### Gene set enrichment analysis
options(stringsAsFactors = F)
library(clusterProfiler) 
library(org.Hs.eg.db)
library(ReactomePA)
library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)
library(igraph)

melanet_spg <- readRDS("./data/spinglass/melanet_spg.RData")
melanet_cmt <- readRDS("./data/spinglass/melanet_cmt.RData")
graph_comp <- readRDS("./data/network_processed/graph_comp.RData")
snv_intgr <- readRDS("./data/tcga_data_processed/snv_intgr.RData")
cnv_intgr <- readRDS("./data/tcga_data_processed/cnv_intgr.RData")
samplePartition <- readRDS("./data/sample_partition.RData")

lengthMat <- matrix(nrow = length(melanet_cmt), ncol = 2, dimnames = list(1:length(melanet_cmt), c("before","after")))

### Reactome pathway enrichment analysis
reactome <- list()
for(i in 1:length(melanet_cmt)){
  gene <- melanet_cmt[[i]]
  lengthMat[i,1] <- length(unique(gene)) 
  gene.df <- bitr(gene,
                  fromType = "SYMBOL",
                  toType = c("ENTREZID"),
                  OrgDb = org.Hs.eg.db)
  lengthMat[i,2] <- length(unique(gene.df$ENTREZID))
  reactome[[i]] <- enrichPathway(gene = gene.df$ENTREZID,
                                 organism = "human",
                                 pvalueCutoff = 0.05,
                                 pAdjustMethod = "BH",
                                 minGSSize = 10,
                                 maxGSSize = 500,
                                 qvalueCutoff = 0.05,
                                 readable = TRUE)
}

emListToDf <- function(em){ # The function to combine all GSEA results.
  emDF <- data.frame()
  for(i in 1:length(em)){
    df <- data.frame()
    if(dim(em[[i]])[1] != 0){
      df <- as.data.frame(em[[i]])
    }
    emDF <- rbind(emDF,df)
  }
  row.names(emDF) <- 1:nrow(emDF)
  return(emDF)
}

reactomeDF <- emListToDf(reactome)
reactomeDF <- reactomeDF[,c("ID","Description","BgRatio")]
reactomeDF <- unique(reactomeDF) # Remove duplicates
row.names(reactomeDF) <- reactomeDF$ID
reactomeDF$BgRatio <- unlist(lapply(strsplit(reactomeDF$BgRatio, split = "/"), function(x){x[1]})) # The bg of each pathway

RA_pathways_26clusters <- read.csv("./data/gsea/RA_pathways_26clusters.csv", sep = ";") # The category of each pathway
row.names(RA_pathways_26clusters) <- RA_pathways_26clusters$Reactome_ID
categories <- RA_pathways_26clusters[,grep("category",colnames(RA_pathways_26clusters))]

numMat <- matrix(data = 0, nrow = 26, ncol = length(melanet_cmt), dimnames = list(substr(colnames(categories), 10, nchar(categories)), 1:length(melanet_cmt))) 

emNum_cg <- data.frame(category = substr(colnames(categories), 10, nchar(categories)), emNum_cg = 0)
for(i in 1:26){
  pathways <- c()
  for(j in 1:length(melanet_cmt)){
    if(dim(reactome[[j]])[1] != 0){
      for(k in 1:dim(reactome[[j]])[1]){
        pathway = as.data.frame(reactome[[j]])$ID[k]
        if(pathway %in% row.names(categories) == T){
          if(categories[pathway,i] == 1){
            pathways <- c(pathways, pathway)
            numMat[i,j] <- numMat[i,j] + 1 # Category’s number of significantly enriched pathways for the community
          }
        }
      }
    }
  }
  emNum_cg[i,2] <- length(unique(pathways))
}


numMat <- numMat[which(apply(numMat, 1, sum) != 0),]
emNum_cg <- emNum_cg[which(emNum_cg$emNum_cg != 0),] # Total number of pathways for the category
col_fun = colorRamp2(c(0,max(numMat)), c("#F4F4F4", "#68B0AB"))
emNum_cmt <- apply(numMat, 2, sum) # Total number of enriched pathways for the community
bottomAnno <- HeatmapAnnotation(emNum_cmt = anno_text(emNum_cmt))

ht_ht <- ceiling(0.5*dim(numMat)[1]) 
ht_wd <- ceiling(0.5*dim(numMat)[2])

categories_mat <- as.matrix(categories)
categories_mat <- categories_mat[,paste("category_",row.names(numMat),sep = "")]

totalNum_cg <- apply(categories_mat, 2, sum) # Category’s number of significantly enriched pathways for all communities
# rowAnno_left <- rowAnnotation(totalNum_cg = anno_barplot(totalNum_cg), show_annotation_name = F)

prop <- emNum_cg[,2] / totalNum_cg * 100
prop_rev <- 100 - prop
prop_rev <- ifelse(prop_rev < 0, 0, prop_rev)
rowAnno_right <- rowAnnotation(prop = anno_barplot(cbind(prop, prop_rev), gp = gpar(fill = c(9,8))), 
                               anno = anno_text(paste(emNum_cg[,2], totalNum_cg, sep = "/")), 
                               show_annotation_name = F)

# axis_param = list(at = c(0,50,100), labels = c("0","50%","100%"))
cmtNodesNum <- sapply(1:length(melanet_spg), function(x){vcount(induced_subgraph(graph = graph_comp, v = which(melanet_spg$membership==x)))})
cmtEdgesNum <- sapply(1:length(melanet_spg), function(x){ecount(induced_subgraph(graph = graph_comp, v = which(melanet_spg$membership==x)))})
nodes_col_fun <- colorRamp2(c(0,max(cmtNodesNum)), c("white", "#F0E442")) 
edges_col_fun <- colorRamp2(c(0,max(cmtEdgesNum)), c("white", "#0072B2"))
topAnno <- HeatmapAnnotation('# of nodes' = cmtNodesNum, '# of edges' = cmtEdgesNum, annotation_name_side = "left", col = list('# of nodes' = nodes_col_fun, '# of edges' = edges_col_fun))

### Gene set enrichment analysis of the network communities.
ht <- Heatmap(numMat,
              row_names_side = "left",
              column_names_side = "top",
              width = unit(ht_wd,"cm"),
              height = unit(ht_ht,"cm"),
              cluster_columns = F,
              show_row_dend = F,
              show_column_dend = F,
              col = col_fun,
              heatmap_legend_param = list(title = "# of pathwahys"),
              # left_annotation = rowAnno_left,
              right_annotation = rowAnno_right,
              top_annotation = topAnno,
              bottom_annotation = bottomAnno,
              row_order = row.names(numMat)[order(totalNum_cg, decreasing = T)],
              cell_fun = function(j, i, x, y, width, height, fill){
                if(numMat[i,j] != 0){
                  grid.text(numMat[i,j], x = x, y = y,
                            gp = gpar(fontsize = 10, col="black"))
                }
              }
)
pdf("./figure/gsea_fig/cmt_category_num.pdf", width = 21, height = 29.7)
draw(ht, merge_legends = TRUE)
dev.off() 

### The interacting genes
# em: enrichment result
# cmt: community
# category: Reactome category
geneOverlap <- function(em, cmt, category){
  overlapGeneList <- list()
  emGenes <- as.data.frame(em[[cmt]])$geneID # enriched genes
  if(length(emGenes) != 0){
    emGeneList <- list() 
    for(i in 1:length(emGenes)){
      emGeneList[[i]] <- unlist(strsplit(emGenes[i], split = "/"))
    }
    names(emGeneList) <- as.data.frame(em[[cmt]])$ID 
    
    # SHAP values
    shapFhtseq <- read.csv(paste("./data/python_related/shap/cmt", cmt, "reshtseq.csv", sep = ""), check.names = F, header = F) 
    shapFmethy <- read.csv(paste("./data/python_related/shap/cmt", cmt, "resmethy.csv", sep = ""), check.names = F, header = F)
    
    shapFhtseq <- shapFhtseq[order(shapFhtseq[,2], decreasing = T),]
    shapFmethy <- shapFmethy[order(shapFmethy[,2], decreasing = T),]
    
    # Top20 SHAP genes
    shapFhtseq <- shapFhtseq[1:min(20,nrow(shapFhtseq)),1]
    shapFmethy <- shapFmethy[1:min(20,nrow(shapFmethy)),1]
    shapFhtseq <- gsub('["]', '', shapFhtseq)
    shapFmethy <- gsub('["]', '', shapFmethy)
    shapFhtseq <- data.frame(gene = shapFhtseq, rank = 1:length(shapFhtseq),row.names = shapFhtseq)
    shapFmethy <- data.frame(gene = shapFmethy, rank = 1:length(shapFmethy),row.names = shapFmethy)
    
    j = 1
    for(i in 1:length(emGeneList)){
      if(names(emGeneList)[i] %in% row.names(categories) == F){
        next
      } 
      if(categories[names(emGeneList)[i],category] == 0){
        next
      }
      htseq <- emGeneList[[i]][which(emGeneList[[i]] %in% shapFhtseq[,1])]
      methy <- emGeneList[[i]][which(emGeneList[[i]] %in% shapFmethy[,1])] 
      
      if(length(htseq) !=0 | length(methy) != 0){
        Genes <- c()
        if(length(htseq) != 0){
          Genes <- append(Genes, paste0('exp_',htseq))
        }
        if(length(methy) != 0){
          Genes <- append(Genes, paste0('mty_',methy))
        }
        overlapGeneList[[j]] <- Genes
        names(overlapGeneList)[j] <- names(emGeneList)[i]  
        j = j+1
      }
    }
  }
  return(overlapGeneList)
}

rankOverlap <- function(em, cmt, category){
  overlapRankList <- list() 
  emGenes <- as.data.frame(em[[cmt]])$geneID 
  if(length(emGenes) != 0){
    emGeneList <- list() 
    for(i in 1:length(emGenes)){
      emGeneList[[i]] <- unlist(strsplit(emGenes[i], split = "/"))
    }
    names(emGeneList) <- as.data.frame(em[[cmt]])$ID 
    
    shapFhtseq <- read.csv(paste("./data/python_related/shap/cmt", cmt, "reshtseq.csv", sep = ""), check.names = F, header = F)
    shapFmethy <- read.csv(paste("./data/python_related/shap/cmt", cmt, "resmethy.csv", sep = ""), check.names = F, header = F)

    shapFhtseq <- shapFhtseq[order(shapFhtseq[,2], decreasing = T),]
    shapFmethy <- shapFmethy[order(shapFmethy[,2], decreasing = T),]
    
    shapFhtseq <- shapFhtseq[1:min(20,nrow(shapFhtseq)),1]
    shapFmethy <- shapFmethy[1:min(20,nrow(shapFmethy)),1]
    shapFhtseq <- gsub('["]', '', shapFhtseq)
    shapFmethy <- gsub('["]', '', shapFmethy)
    shapFhtseq <- data.frame(gene = shapFhtseq, rank = 1:length(shapFhtseq),row.names = shapFhtseq)
    shapFmethy <- data.frame(gene = shapFmethy, rank = 1:length(shapFmethy),row.names = shapFmethy)
    
    j = 1
    for(i in 1:length(emGeneList)){
      if(names(emGeneList)[i] %in% row.names(categories) == F){
        next
      } 
      if(categories[names(emGeneList)[i],category] == 0){
        next
      }
      htseq <- emGeneList[[i]][which(emGeneList[[i]] %in% shapFhtseq[,1])]
      methy <- emGeneList[[i]][which(emGeneList[[i]] %in% shapFmethy[,1])]
      
      rank_htseq <- shapFhtseq[htseq,2]
      rank_methy <- shapFmethy[methy,2]
      
      if(length(htseq) !=0 | length(methy) != 0){
        Ranks <- c()
        if(length(htseq) != 0){
          Ranks <- append(Ranks, rank_htseq)
        }
        if(length(methy) != 0){
          Ranks <- append(Ranks, rank_methy)
        }
        overlapRankList[[j]] <- Ranks
        names(overlapRankList)[j] <- names(emGeneList)[i]  
        j = j+1
      }
    }
  }
  return(overlapRankList)
}

melanoma_map_nodes <- read.csv("./data/melanoma_network/expanded_network_nodes.csv", header = T, check.names = F)
melanoma_map_edges <- read.csv("./data/melanoma_network/expanded_network_edges.csv", header = T, check.names = F)
core_map_nodes <- read.csv("./data/melanoma_network/core_network_nodes.csv", header = T, check.names = F)
core_map_edges <- read.csv("./data/melanoma_network/core_network_edges.csv", header = T, check.names = F)

### Drugs in the melanoma network
drugs <- melanoma_map_nodes[melanoma_map_nodes$class == "DRUG",c("id","name")]
targetIDs <- melanoma_map_edges[melanoma_map_edges$source %in% drugs$id, c("source","target")]
targetSymbols <- melanoma_map_nodes[melanoma_map_nodes$id %in% targetIDs$target,c("id","hgnc_symbol")]
targetSymbols <- targetSymbols[which(targetSymbols$hgnc_symbol != ""),]
targets <- merge(targetIDs, targetSymbols, by.x = "target", by.y = "id")
drugs <- merge(drugs, targets, by.x = "id", by.y = "source")
drugs <- drugs[which(!duplicated(drugs)),]

### Genes in the melanoma core network
curated_genes <- core_map_nodes$hgnc_symbol
curated_genes <- curated_genes[which(curated_genes != "")]
curated_genes <- curated_genes[which(!duplicated(curated_genes))] 


# category: Reactome category
gene_list <- list()
c = 1
for(category in 1:26){
  print(category)
  overlapGeneList <- list()
  overlapRankList <- list()
  for(i in 1:length(melanet_cmt)){
    overlapGeneList[[i]] <- geneOverlap(reactome, i, category)
    overlapRankList[[i]] <- rankOverlap(reactome, i, category)
  }
  
  if(all(lengths(overlapGeneList) == 0)){
    print("There are no interacting genes in this category.")
    next
  }
  
  pathways <- c() 
  genes <- c() 
  for(i in 1:length(melanet_cmt)){
    if(lengths(overlapGeneList)[i] != 0){
      pathways <- append(pathways, names(overlapGeneList[[i]]))
      genes <- append(genes, as.character(unlist(overlapGeneList[[i]])))
    }
  }
  pathways <- unique(pathways)
  genes <- unique(genes)
  
  gene_list[[c]] <- genes 
  names(gene_list)[c] <- colnames(categories)[category]
  c = c + 1
  
  cmtMat <- matrix(data = "NA", nrow = length(pathways), ncol = length(genes), dimnames = list(pathways,genes)) 
  rankMat <- matrix(data = "NA", nrow = length(pathways), ncol = length(genes),dimnames = list(pathways,genes)) 
  
  for(i in 1:length(melanet_cmt)){
    if(lengths(overlapGeneList)[i] != 0){
      for(j in 1:length(overlapGeneList[[i]])){
        rname <- names(overlapGeneList[[i]])[j]
        cnames <- overlapGeneList[[i]][[j]]
        for(k in 1:length(cnames)){
          cname <- cnames[k]
          cmtMat[rname,cname] = as.character(i)
          rankMat[rname,cname] = as.character(overlapRankList[[i]][[j]][k])
        }
      }
    }
  }
  
  col_fun <- c("NA" = "#D9D9D9",
               "3" = "#A6CEE3",
               "5" = "#1F78B4",
               "7" = "#B2DF8A",
               "8" = "#33A02C",
               "10" = "#FB9A99",
               "12" = "#E31A1C",
               "13" = "#FDBF6F",
               "14" = "#FF7F00",
               "17" = "#CAB2D6",
               "18" = "#6A3D9A",
               "20" = "#FFFF99",
               "21" = "#B15928")
  category_col_fun <- c("exp" = "#FFC7C7", "mty" = "#68B0AB") 
  druggable_col_fun <- c("FALSE" = "#D9ECF2", "TRUE" = "#F56A79")
  
  topAnno <- data.frame(genes = substr(genes,5,nchar(genes)), category = substr(genes,1,3), row.names = genes) # 基因名称注释
  topAnno <- topAnno[order(topAnno$category),]
  
  snv_cluster1 <- sapply(topAnno$genes, function(x){
    ifelse(x %in% colnames(snv_intgr), sum(snv_intgr[which(samplePartition$cluster == 1), x]) / length(which(samplePartition$cluster == 1)) * 100, 0)
  }) 
  snv_cluster2 <- sapply(topAnno$genes, function(x){
    ifelse(x %in% colnames(snv_intgr), sum(snv_intgr[which(samplePartition$cluster == 2), x]) / length(which(samplePartition$cluster == 2)) * 100, 0)
  })
  snv_cluster3 <- sapply(topAnno$genes, function(x){
    ifelse(x %in% colnames(snv_intgr), sum(snv_intgr[which(samplePartition$cluster == 3), x]) / length(which(samplePartition$cluster == 3)) * 100, 0)
  })
  
  cnv_cluster1 <- sapply(topAnno$genes, function(x){
    ifelse(x %in% colnames(cnv_intgr), length(which(cnv_intgr[which(samplePartition$cluster == 1), x] != 0)) / length(which(samplePartition$cluster == 1)) * 100, 0)
  })
  cnv_cluster2 <- sapply(topAnno$genes, function(x){
    ifelse(x %in% colnames(cnv_intgr), length(which(cnv_intgr[which(samplePartition$cluster == 2), x] != 0)) / length(which(samplePartition$cluster == 2)) * 100, 0)
  }) 
  cnv_cluster3 <- sapply(topAnno$genes, function(x){
    ifelse(x %in% colnames(cnv_intgr), length(which(cnv_intgr[which(samplePartition$cluster == 3), x] != 0)) / length(which(samplePartition$cluster == 3)) * 100, 0)
  })
  
  snv_col_fun <- colorRamp2(c(0,max(snv_cluster1, snv_cluster2, snv_cluster3)), c( "white", "#F0E442")) # 突变比率颜色
  cnv_col_fun <- colorRamp2(c(0,max(cnv_cluster1, cnv_cluster2, cnv_cluster3)), c("white", "#0072B2")) # 拷贝数变异比率颜色
  
  druggable <-  sapply(topAnno$genes,function(x){x %in% drugs$hgnc_symbol})
  drug_name <- rep("", length(topAnno$genes))
  drug_name <- sapply(topAnno$genes, function(x){paste0(drugs[which(drugs$hgnc_symbol==x),"name"], collapse = "+")})
  
  
  if(ncol(cmtMat) == 1 | nrow(cmtMat) == 1){
    print("There is only one interacting gene in this category")
  }else{
    cmtMat <- cmtMat[,row.names(topAnno)] 
    rankMat <- rankMat[,row.names(topAnno)]
  }
  rowLabels <- reactomeDF[row.names(cmtMat), "Description"]
  bg <- as.numeric(reactomeDF[row.names(cmtMat), "BgRatio"])
  rowAnno <- rowAnnotation(bg = anno_barplot(bg), show_annotation_name = F) 
  
  ht_ht <- ceiling(0.5*dim(cmtMat)[1])
  ht_wd <- ceiling(0.5*dim(cmtMat)[2])
  
  ### Biological interpretation of the top-ranking genomic features in the network communities
  ht<-Heatmap(cmtMat,
              width = unit(ht_wd,"cm"),
              height = unit(ht_ht,"cm"),
              col = col_fun, 
              row_names_side = "left",
              row_names_gp = gpar(fontsize = 10),
              row_names_centered = F,
              row_labels = rowLabels,
              row_names_max_width = unit(20, "cm"),
              row_order = row.names(cmtMat)[order(bg,decreasing = T)],
              left_annotation = rowAnno,
              column_names_side = "top",
              column_names_gp = gpar(col = ifelse(topAnno$genes %in% curated_genes,"red","black"),fontsize = 10), 
              column_names_centered = F,
              column_names_rot = 90,
              column_labels = topAnno$genes,
              top_annotation = HeatmapAnnotation(category =topAnno$category, snv_cluster1 = snv_cluster1, snv_cluster2 = snv_cluster1, snv_cluster3 = snv_cluster3, cnv_cluster1 = cnv_cluster1, cnv_cluster2 = cnv_cluster2, cnv_cluster3 = cnv_cluster3, col = list(category = category_col_fun, snv_cluster1 = snv_col_fun, snv_cluster2 = snv_col_fun, snv_cluster3 = snv_col_fun, cnv_cluster1 = cnv_col_fun, cnv_cluster2 = cnv_col_fun, cnv_cluster3 = cnv_col_fun), show_annotation_name = c(F,T,T,T,T,T,T), show_legend = c(T,T,F,F,T,F,F), annotation_legend_param = list(snv_cluster1 = list(title = "snv(%)"), cnv_cluster1 = list(title = "cnv(%)")), annotation_name_side = "left"),
              bottom_annotation = HeatmapAnnotation(druggable = druggable, drug_name = anno_text(drug_name, gp = gpar(fontsize = 10), rot = 45), col = list(druggable = druggable_col_fun), show_annotation_name = F),
              heatmap_legend_param = list(title = "cmt"),
              cell_fun = function(j, i, x, y, width, height, fill){
                if(rankMat[i,j]!="NA"){
                  grid.text(rankMat[i,j], x = x, y = y,
                            gp = gpar(fontsize = 10, col="black")) 
                }
              },
              show_row_dend = F,
              cluster_columns = F,
  )
  pdf(paste("./figure/gsea_fig/", colnames(categories)[category],".pdf",sep = ""), width = 21, height = 29.7)
  draw(ht, merge_legends = TRUE)
  dev.off()
}

### Interactome profiles of the top-ranking omics features. 
nodes_comp <- data.frame(symbol = V(graph_comp)$name, cmt = melanet_spg$membership)
edges_comp <- as.data.frame(as_edgelist(graph_comp))
colnames(edges_comp) <- c("source_symbol", "target_symbol")
edges_comp$source_id <- sapply(edges_comp$source_symbol, function(x){paste(melanoma_map_nodes[which(melanoma_map_nodes$hgnc_symbol == x),"id"],collapse = ";")})
edges_comp$target_id <- sapply(edges_comp$target_symbol, function(x){paste(melanoma_map_nodes[which(melanoma_map_nodes$hgnc_symbol == x),"id"],collapse = ";")})
edges_comp$Interactiontype <- mapply(function(x,y){paste(sort(unique(melanoma_map_edges[which(melanoma_map_edges$source %in% unlist(strsplit(x, split = ";")) & melanoma_map_edges$target %in% unlist(strsplit(y, split = ";"))), "Interactiontype"])), collapse = ";")}, edges_comp$source_id, edges_comp$target_id)
edges_comp$type <- mapply(function(x,y){paste(sort(unique(melanoma_map_edges[which(melanoma_map_edges$source %in% unlist(strsplit(x, split = ";")) & melanoma_map_edges$target %in% unlist(strsplit(y, split = ";"))), "type"])), collapse = ";")}, edges_comp$source_id, edges_comp$target_id)

edges_comp$Source <- ifelse(edges_comp$type == "DATABASE", "DATABASE", "Literature")
edges_comp$Interationtype_processed <- edges_comp$Interactiontype
edges_comp$Interationtype_processed[which(edges_comp$Interationtype_processed %in% c("interaction;stimulation"))] <- "stimulation"
edges_comp$Interationtype_processed[which(edges_comp$Interationtype_processed %in% c("inhibition;interaction"))] <- "inhibition"

saveRDS(nodes_comp, "./data/network_processed/nodes_comp.RData")
saveRDS(edges_comp, "./data/network_processed/edges_comp.RData")

for(i in 1:length(gene_list)){
  print(i)
  genes <- gene_list[[i]]
  geneDf <- data.frame(gene = substr(genes, 5, nchar(genes)), category = substr(genes, 1, 3), row.names = genes)
  gene_exp <- geneDf[which(geneDf$category == "exp"),"gene"] # 属于基因表达的重叠基因
  gene_mty <- geneDf[which(geneDf$category == "mty"),"gene"] # 属于甲基化的重叠基因
  gene_union <- union(gene_exp, gene_mty) # 两类基因并集
  gene_intersect <- intersect(gene_exp, gene_mty) # 两类基因交集
  
  neighbrs <- neighborhood(graph_comp, order = 1, nodes = gene_union)
  neighbrs <- setdiff(names(unlist(neighbrs)), gene_union)
  
  subgraph_comp <- induced_subgraph(graph_comp, v = union(gene_union, neighbrs))
  
  category <- ifelse(V(subgraph_comp)$name %in% gene_intersect, "both", ifelse(V(subgraph_comp)$name %in% gene_exp, "exp", ifelse(V(subgraph_comp)$name %in% gene_mty, "mty", "neighbor")))
  ifcurated <- ifelse(V(subgraph_comp)$name %in% curated_genes, "TRUE", "FALSE") # 是否是核心网络里的基因
  
  nodeDf <- data.frame(gene = V(subgraph_comp)$name, category = category, ifcurated = ifcurated) # 节点信息
  write.csv(nodeDf, paste("./figure/interaction_profile/nodeInfo/", "nodeDf_", substr(names(gene_list)[[i]], 10, nchar(names(gene_list)[[i]])), ".csv", sep = ""), row.names = F) # 保存节点信息为csv
  
  edgeDf <- as.data.frame(as_edgelist(subgraph_comp))
  colnames(edgeDf) <- c("source_symbol","target_symbol")
  edgeDf$source_id <- sapply(edgeDf$source_symbol, function(x){paste(melanoma_map_nodes[which(melanoma_map_nodes$hgnc_symbol == x),"id"],collapse = ";")})
  edgeDf$target_id <- sapply(edgeDf$target_symbol, function(x){paste(melanoma_map_nodes[which(melanoma_map_nodes$hgnc_symbol == x),"id"],collapse = ";")})
  edgeDf$Interactiontype <- mapply(function(x,y){paste(sort(unique(melanoma_map_edges[which(melanoma_map_edges$source %in% unlist(strsplit(x, split = ";")) & melanoma_map_edges$target %in% unlist(strsplit(y, split = ";"))), "Interactiontype"])), collapse = ";")}, edgeDf$source_id, edgeDf$target_id)
  edgeDf$type <- mapply(function(x,y){paste(sort(unique(melanoma_map_edges[which(melanoma_map_edges$source %in% unlist(strsplit(x, split = ";")) & melanoma_map_edges$target %in% unlist(strsplit(y, split = ";"))), "type"])), collapse = ";")}, edgeDf$source_id, edgeDf$target_id)
  
  edgeDf$Source <- ifelse(edgeDf$type == "DATABASE", "DATABASE", "Literature")
  edgeDf$Interationtype_processed <- edgeDf$Interactiontype
  edgeDf$Interationtype_processed[which(edgeDf$Interationtype_processed %in% c("interaction;stimulation"))] <- "stimulation"
  edgeDf$Interationtype_processed[which(edgeDf$Interationtype_processed %in% c("inhibition;interaction"))] <- "inhibition"
  
  write.csv(edgeDf, paste("./figure/interaction_profile/edgeInfo/", "edgeDf_", substr(names(gene_list)[[i]], 10, nchar(names(gene_list)[[i]])), ".csv", sep = ""), row.names = F) # 保存边的信息
  
  Source <- edgeDf$Source
  Interactiontype <- edgeDf$Interationtype_processed
  
  # p <- ggraph(subgraph_comp, layout = "kk") + 
  #   geom_node_point(aes(colour = category, size = 30)) + 
  #   geom_node_text(aes(colour = ifcurated, label = V(subgraph_comp)$name), size = 5, repel = T, show.legend = F) +
  #   geom_edge_link0(aes(edge_colour = Interactiontype, edge_linetype = Source)) +
  #   scale_colour_manual(values = c("exp" = "#FFC7C7", "mty" = "#68B0AB", "both" = "#E69F00", "TRUE" = "red", "FALSE" = "black", "neighbor" = "gray")) +
  #   scale_edge_color_manual(values = c("interaction"="#BEBADA", "inhibition"="#80B1D3", "stimulation"="#FB8072", "inhibition;stimulation"="#FDB462")) +
  #   scale_edge_linetype_manual(values = c("DATABASE" = "dashed", "Literature" = "solid")) +
  #   theme_void()
  # 
  # pdf(paste("./figure/genes_network/", "network_", substr(names(gene_list)[[i]], 10, nchar(names(gene_list)[[i]])), ".pdf", sep = ""), width = 21, height = 29.7/2)
  # print(p)
  # dev.off()
}
a