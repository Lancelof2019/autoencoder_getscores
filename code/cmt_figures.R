##### The integration of the genomic profiles into the network communities
options(stringsAsFactors = F)
library(igraph)
library(graphlayouts)
library(reshape2)
library(ggraph)
library(ggpubr)

graph_comp <- readRDS("./data/network_processed/graph_comp.RData")
melanet_spg <- readRDS("./data/spinglass/melanet_spg.RData")
exp_intgr <- readRDS("./data/tcga_data_processed/exp_intgr.RData")
mty_intgr <- readRDS("./data/tcga_data_processed/mty_intgr.RData")
snv_intgr <- readRDS("./data/tcga_data_processed/snv_intgr.RData")
cnv_intgr <- readRDS("./data/tcga_data_processed/cnv_intgr.RData")
samples <- readRDS("./data/tcga_data_processed/samples.RData")

lapply(1:21, function(i){
  print(i)
  graph_comp_sub <- induced_subgraph(graph = graph_comp, v = which(melanet_spg$membership == i))
  stress <- layout_with_stress(graph_comp_sub)
  V(graph_comp_sub)$x <- stress[,1]
  V(graph_comp_sub)$y <- stress[,2]
  
  circle_list <- lapply(1:vcount(graph_comp_sub), function(j){
    print(j)
    gene <- V(graph_comp_sub)$name[j]
    exp <- NA
    if(gene %in% colnames(exp_intgr)){
      exp <- exp_intgr[,gene]
      exp <- -1 + 2 * (exp - min(exp, na.rm = T))/(max(exp, na.rm = T) - min(exp, na.rm = T))
    }
    mty <- NA
    if(gene %in% colnames(mty_intgr)){
      mty <- mty_intgr[,gene]
      mty <- -1 + 2 * (mty - min(mty, na.rm = T))/(max(mty, na.rm = T) - min(mty, na.rm = T))
    }
    snv <- NA
    if(gene %in% colnames(snv_intgr)){
      snv <- snv_intgr[,gene]
      snv <- ifelse(snv > 0, 1, snv)
    }
    cnv <- NA
    if(gene %in% colnames(cnv_intgr)){
      cnv <- cnv_intgr[,gene]
    }
    node_wide <- data.frame(sample = samples, exp = exp, mty = mty, snv = snv, cnv = cnv, point = 0)
    node_long <- melt(node_wide, id.vars = c("sample"), variable.name = "category", value.name = "value")
    node_long$category <- factor(node_long$category, levels = c("point","cnv","snv","mty","exp"))
    node_long$value <- ifelse(node_long$value>5, 5, node_long$value)
    
    gt_plot <- ggplotGrob(ggplot(node_long, aes(x = category, y = 1, fill = value)) + geom_bar(stat = "identity", width = 0.3, position = "stack") + labs(x = NULL, y = NULL) +  coord_polar(theta = "y") + scale_fill_gradient2(low = "blue", high = "red", mid = "#F7F7F7", midpoint = 0, limits = c(-1,1)) + scale_color_brewer(palette = "Pastel2", direction = -1) + theme(legend.position = "none", panel.background = element_rect(fill  = NA, colour = NA), line = element_blank(), text = element_blank()))
    panel_coords <- gt_plot$layout[gt_plot$layout$name == "panel", ]
    gt_plot[panel_coords$t:panel_coords$b, panel_coords$l:panel_coords$r]
  })
  
  # Convert to annotation
  annot_list <- lapply(1:vcount(graph_comp_sub), function(k) {
    print(k)
    xmin <- V(graph_comp_sub)$x[k] - 0.1
    xmax <- V(graph_comp_sub)$x[k] + 0.1
    ymin <- V(graph_comp_sub)$y[k] - 0.1
    ymax <- V(graph_comp_sub)$y[k] + 0.1
    annotation_custom(
      circle_list[[k]],
      xmin = xmin,
      xmax = xmax,
      ymin = ymin,
      ymax = ymax
    )
  })
  
  # Community
  p <- ggraph(graph_comp_sub, layout = "manual", x = V(graph_comp_sub)$x, y = V(graph_comp_sub)$y) + geom_edge_link(edge_color = "grey66", width = 0.2) + coord_fixed() + theme(panel.background = element_rect(fill = "white", colour = NA))
  
  num_nodes <- vcount(graph_comp_sub)
  num_edges <- ecount(graph_comp_sub)
  
  # Top 5 highest degree nodes
  nodes <- paste(names(sort(igraph::degree(graph_comp_sub), decreasing = T)[1:5]), collapse = ", ")
  
  pdf(paste("./figure/cmt_figures/cmt_",i,".pdf",sep = ""))
  print(ggarrange(Reduce("+", annot_list, p), nrow = 1, ncol = 1, widths = 7, heights = 7, labels = paste("Cmt", i, ": ", num_nodes, " nodes, ", num_edges," edges ", "(", nodes, ")",sep = ""), font.label = list(size = 12), hjust = -0.1, align = "hv"))
  dev.off()
})

# Circle plot of TP53
gene <- "TP53"
exp <- NA
if(gene %in% colnames(exp_intgr)){
  exp <- exp_intgr[,gene]
  exp <- -1 + 2 * (exp - min(exp, na.rm = T))/(max(exp, na.rm = T) - min(exp, na.rm = T))
}
mty <- NA
if(gene %in% colnames(mty_intgr)){
  mty <- mty_intgr[,gene]
  mty <- -1 + 2 * (mty - min(mty, na.rm = T))/(max(mty, na.rm = T) - min(mty, na.rm = T))
}
snv <- NA
if(gene %in% colnames(snv_intgr)){
  snv <- snv_intgr[,gene]
  snv <- ifelse(snv > 0, 1, snv)
}
cnv <- NA
if(gene %in% colnames(cnv_intgr)){
  cnv <- cnv_intgr[,gene]
}
node_wide <- data.frame(sample = samples, exp = exp, mty = mty, snv = snv, cnv = cnv, point = 0)
node_long <- melt(node_wide, id.vars = c("sample"), variable.name = "category", value.name = "value")
node_long$category <- factor(node_long$category, levels = c("point","cnv","snv","mty","exp"))
node_long$value <- ifelse(node_long$value>5, 5, node_long$value)
pdf("./figure/cmt_figures/circle_tp53.pdf")
ggplot(node_long, aes(x = category, y = 1, fill = value)) + geom_bar(stat = "identity", width = 0.3, position = "stack") + labs(x = NULL, y = NULL) +  coord_polar(theta = "y") + scale_fill_gradient2(low = "blue", high = "red", mid = "#F7F7F7", midpoint = 0, limits = c(-1,1)) + scale_color_brewer(palette = "Pastel2", direction = -1) + theme_bw()
dev.off()