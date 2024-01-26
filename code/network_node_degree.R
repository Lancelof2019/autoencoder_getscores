##### The node degree of the network
options(stringsAsFactors = F)
library(ggplot2)
library(igraph)

graph_comp <- readRDS("./data/network_processed/graph_comp.RData")
net_num_nodes <- vcount(graph_comp) # Number of nodes
net_num_edges <- ecount(graph_comp) # Number of edges
net_degree_dist <- data.frame(degree = log10(igraph::degree(graph_comp))) # log10 conversion of node degree
net_degree_table <- as.data.frame(prop.table(table(net_degree_dist))) # Node degree frequency
net_degree_table$net_degree_dist <- as.numeric(as.character(net_degree_table$net_degree_dist))
net_degree_table$Freq <- log10(net_degree_table$Freq) # log10 conversion of frequency

pdf("./figure/network_degree_distribution.pdf", width = 7, height = 7)
ggplot(net_degree_table, aes(x = net_degree_dist, y = Freq)) +
  geom_point() + 
  geom_smooth(method = "lm", color = "blue", se = FALSE) + 
  labs(title = paste("Network", ": ", net_num_nodes, " nodes, ", net_num_edges," edges", sep = ""), x = "Node degree (log10)", y = "Frequency (log10)") + 
  theme(text = element_text(size = 18), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black'))
dev.off()