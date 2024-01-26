##### Distribution of modularity.
options(stringsAsFactors = F)
library(ggplot2)

spg_mod <- readRDS("./data/spinglass/spg_mod.RData")
df_all <- data.frame(mod = spg_mod)
hist_all <- ggplot(data = df_all, aes(x = mod)) +
  geom_histogram(fill = "#C1BFBF") + 
  theme(text = element_text(size = 20), axis.title = element_text(), plot.title = element_text(hjust = 0.5, face = "bold"), panel.grid = element_blank(), panel.background = element_rect(fill = 'transparent', color = 'black')) +
  labs(x = "Modularity", y = "Frequency", title = "Histogram of modularity (1150 runs)")
pdf("./figure/spinglass_modularity_distribution_all.pdf", width = 21 / 3, height = 29.7 / 4)
hist_all
dev.off()
