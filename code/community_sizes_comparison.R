##### Community sizes of the top ten simulations ranked by modularity values.
options(stringsAsFactors = F)
library(igraph)
library(ggplot2)
library(reshape2)
library(tidyverse)

graph_comp <- readRDS("./data/network_processed/graph_comp.RData")
spg <- readRDS("./data/spinglass/spg.RData")
spg_mod <- readRDS("./data/spinglass/spg_mod.RData")
melanet_spg <- readRDS("./data/spinglass/melanet_spg.RData")

# keep the simulations of which the communities are fully connected
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

spg_top10mod <- spg[full_conn_runs[1:10]]
plot_csize <- data.frame(crank = factor(1:22, levels = 22:1))
for(i in 1:10){
  print(spg_top10mod[[i]]$modularity)
}

for(i in 1:length(spg_top10mod)){
  run <- paste("run", full_conn_runs[i], sep = "")
  plot_csize[,run] <- NA
  csize <- sort(lengths(communities(spg_top10mod[[i]])), decreasing = T)
  plot_csize[1:length(csize),run] <- csize
}

pdf("./figure/spinglass_top10mod_cmtsize.pdf", width = 21, height = 29.7 / 2)
ggplot(melt(plot_csize, id.vars = "crank"), aes(x = variable, y = value, fill = crank)) + 
  geom_bar(stat = "identity", position = "stack", width = 0.5, col = "black") + 
  labs(x = "Simulation", y = "Community size", title = "Community size distribution (fully connected and top 10 modularity)") + 
  theme_classic(base_size = 24) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"), legend.position = c(0.95,0.75)) +
  labs(fill = "Rank") +
  scale_y_continuous(breaks = seq(0,5200,200)) +
  guides(fill = guide_legend(ncol = 1)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupA = cumsum(run32)) %>% mutate(GroupB = cumsum(run700)), aes(x = 1.25, xend = 1.75, y = GroupA, yend = GroupB)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupB = cumsum(run700)) %>% mutate(GroupC = cumsum(run817)), aes(x = 2.25, xend = 2.75, y = GroupB, yend = GroupC)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupC = cumsum(run817)) %>% mutate(GroupD = cumsum(run335)), aes(x = 3.25, xend = 3.75, y = GroupC, yend = GroupD)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupD = cumsum(run335)) %>% mutate(GroupE = cumsum(run102)), aes(x = 4.25, xend = 4.75, y = GroupD, yend = GroupE)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupE = cumsum(run102)) %>% mutate(GroupF = cumsum(run161)), aes(x = 5.25, xend = 5.75, y = GroupE, yend = GroupF)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupF = cumsum(run161)) %>% mutate(GroupG = cumsum(run485)), aes(x = 6.25, xend = 6.75, y = GroupF, yend = GroupG)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupG = cumsum(run485)) %>% mutate(GroupH = cumsum(run260)), aes(x = 7.25, xend = 7.75, y = GroupG, yend = GroupH)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupH = cumsum(run260)) %>% mutate(GroupI = cumsum(run511)), aes(x = 8.25, xend = 8.75, y = GroupH, yend = GroupI)) +
  geom_segment(data = plot_csize %>% arrange(by = desc(crank)) %>% mutate(GroupI = cumsum(run511)) %>% mutate(GroupJ = cumsum(run562)), aes(x = 9.25, xend = 9.75, y = GroupI, yend = GroupJ))
dev.off()