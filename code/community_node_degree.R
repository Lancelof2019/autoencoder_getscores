### The node degree distribution of the network communities
library(ggplot2)
library(ggpubr)

graph_comp <- readRDS("./data/network_processed/graph_comp.RData")
melanet_spg <- readRDS("./data/spinglass/melanet_spg.RData")

hist_list <- lapply(1:length(melanet_spg), function(i){
  print(i)
  graph_comp_sub <- induced_subgraph(graph = graph_comp, v = which(melanet_spg$membership == i)) 
  num_nodes <- vcount(graph_comp_sub)
  num_edges <- ecount(graph_comp_sub)
  degree_dist <- data.frame(degree = log10(igraph::degree(graph_comp_sub)))
  degree_table <- as.data.frame(prop.table(table(degree_dist)))
  degree_table$degree_dist <- as.numeric(as.character(degree_table$degree_dist))
  degree_table$Freq <- log10(degree_table$Freq)
  
  ggplot(degree_table, aes(x = degree_dist, y = Freq)) + 
    geom_point() + 
    geom_smooth(method = "lm", color = "blue", se = FALSE) + 
    labs(title = paste("Cmt", i, ": ", num_nodes, " nodes, ", num_edges," edges", sep = ""), x = "Node degree (log10)", y = "Frequency (log10)") + 
    theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) + 
    scale_y_continuous(limits = c(-3,0)) + 
    scale_x_continuous(limits = c(0, log10(max(igraph::degree(graph_comp)))))
  
})

pdf("./figure/community_degree_distribution.pdf", width = 21, height = 29.7)
ggarrange(plotlist = hist_list[1:21], nrow = 6, ncol = 4, widths = 4.5, heights = 4.5)
dev.off()