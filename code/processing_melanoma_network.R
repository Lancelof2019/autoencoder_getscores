##### The melanoma network and network community detection
options(stringsAsFactors = F)
library(igraph)

### The melanoma network
melanoma_map_nodes <- read.csv("./data/melanoma_network/expanded_network_nodes.csv", header = T, check.names = F)
melanoma_map_edges <- read.csv("./data/melanoma_network/expanded_network_edges.csv", header = T, check.names = F)
core_map_nodes <- read.csv("./data/melanoma_network/core_network_nodes.csv", header = T, check.names = F)
core_map_edges <- read.csv("./data/melanoma_network/core_network_edges.csv", header = T, check.names = F)


### The processing of melanoma expanded network
nodes <- melanoma_map_nodes[,c("id","hgnc_symbol")]
edges <- melanoma_map_edges[,c("source","target")]
nodes <- nodes[which(nodes$hgnc_symbol != ""),] 
nodes <- nodes[which(!duplicated(nodes)),] 
edges <- edges[which(!duplicated(edges)),]
edges <- merge(x = edges, y = nodes, by.x = "source", by.y = "id")
colnames(edges)[3] <- "source_symbol"
edges <- merge(x = edges, y = nodes, by.x = "target", by.y = "id")
colnames(edges)[4] <- "target_symbol"
edges <- edges[,c("source_symbol","target_symbol")]
edges <- edges[which(!duplicated(edges)),]
nodes <- data.frame(hgnc_symbol = nodes$hgnc_symbol)
nodes <- unique(nodes)

graph <- graph_from_data_frame(d = edges, directed = T, vertices = nodes)
is_connected(graph) 
components <- decompose(graph, min.vertices = 100) 
graph_comp <- components[[1]] # Take the maximum connected subgraph
saveRDS(graph_comp, "./data/network_processed/graph_comp.RData")

### Network community detection
spg <- list() # List of spinglass simulations
spg_mod <- numeric() # List of modularity simulations
for (n in c(50, 100, 1000)){
  for (k in 1:n){
    print(Sys.time())
    print(count)
    spg[[count]] <- cluster_spinglass(graph_comp,
                                      spins = 25,
                                      weights = NULL)
    spg_mod[count] <- spg[[count]]$modularity
    count = count + 1
    print(Sys.time())
  }
}

saveRDS(spg, "./data/spinglass/spg.RData")
saveRDS(spg_mod, "./data/spinglass/spg_mod.RData")

# Select the fully connected partitions
full_conn_runs <- numeric() 
for(ind in order(spg_mod, decreasing = T)){
  melanet_spg <- spg[[ind]]
  full_connected <- T
  for(i in 1:length(melanet_spg)){
    graph_comp_sub <- induced_subgraph(graph = graph_comp, v = which(melanet_spg$membership == i)) 
    if(is.connected(graph_comp_sub) == F){
      full_connected <- F
    }
  }
  if(full_connected){
    full_conn_runs <- append(full_conn_runs, ind)
  }
}

# The fully connected partition with the maximum modularity 
spg_ind <- full_conn_runs[1]
melanet_spg <- spg[[spg_ind]]
saveRDS(melanet_spg, "./data/spinglass/melanet_spg.RData")
melanet_cmt <- as.list(communities(melanet_spg)) # Melanoma network communities
saveRDS(melanet_cmt, "./data/spinglass/melanet_cmt.RData")

