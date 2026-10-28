# Facebook Page Classification with an MLP Baseline and a Two-Layer GCN

**COMP3710 individual project | Levi Zheng | Student number: 48825634 | GitHub: Sharpshesho**  
**Selected task:** Multi-layer GNN, Normal difficulty. **Results recorded:** 28-29 October 2026.

This project builds a prototype assistant for a social-network analyst who needs to classify Facebook pages as Company, Government, Politician or TV show. It investigates whether graph connections improve classification compared with each page's own 128 features, and whether confidence scores help identify predictions that should be reviewed. The second question concerns the trade-off between automated coverage, remaining mistakes and review workload.

The feature-only MLP predicts from each page's input vector. The GCN combines those features with information aggregated through two graph-convolution layers, allowing neighbours up to two hops away to influence a prediction. Both produce four class scores. After training, a separate rule selected with validation data accepts predictions with sufficiently high maximum softmax scores and flags the others. The experiment simulates this decision using existing labels; actual human-review performance is not measured.

## Contents

- [Written feasibility review](#written-feasibility-review)
- [Artificial Intelligence Usage Disclosure](#artificial-intelligence-usage-disclosure)
- [References](#references)

## Written feasibility review

*Proposal-stage review, revised 28 October 2026. The planned confidence and error analyses described here are completed in the later results sections of the final report.*

**User need, scope and acceptance criteria.** The intended user is a social-network analyst. The prototype classifies pages into four supplied categories and investigates graph context and confidence. The criteria are to: (1) compare MLP and GCN accuracy, macro-F1, AUROC and per-class precision/recall on identical held-out pages; (2) inspect confidence versus correctness, including confident mistakes; (3) select a validation cutoff and report test accepted accuracy and percentage flagged; and (4) measure training time and inference latency within the resource budget.

**Model choice and course concepts.** The Normal-difficulty task uses the modified Gayanku archive with 128 features per page [1]. An MLP provides a simple baseline. The two-layer GCN applies neighbourhood aggregation [2, 3]. Both use 64 hidden units, ReLU, dropout 0.5, cross-entropy and Adam in PyTorch. The design applies supervised loss minimisation, backpropagation, regularisation, validation-based selection and graph homophily. A stratified 60/20/20 split with seed 42 is held fixed. In this transductive setting, features and connections are available for all nodes, while only training labels enter the optimisation loss.

**Preliminary feasibility evidence.** The audit found 22,470 pages, finite features, valid edge indices and disjoint splits. Five-epoch smoke runs completed for both models. By this review revision, six full CPU runs had also completed: 200 epochs per model for training seeds 42, 43 and 44. MLP mean test accuracy was 80.04% and macro-F1 0.7808, with a 3.06-second mean training loop; GCN achieved 92.53% and 0.9200 in 22.29 seconds. These established that the planned local CPU experiments were practical. Time includes validation and checkpoint saves, excluding loading, final testing and plotting.

**Risks, budget and fallback.** The planned budget was at most 200 epochs or 30 minutes per run; measured runs met it. The code runs exactly 200 epochs and has no automatic 30-minute timeout. Risks include unequal class frequencies, misleading neighbours and overconfidence. The next planned work was to select validation cutoffs, freeze them for testing, profile inference, interpret embeddings and inspect 3-5 mistakes. The fallback was the MLP if graph benefits did not justify cost; smaller hidden layers or GPU resources were options for larger experiments.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with explanations of graph learning and evaluation concepts, Python implementation and debugging, data-audit and reporting code, result checks, and drafting this README and feasibility review. AI assistance included substantive code and document development.

Verification used the local data audit, five-epoch MLP/GCN smoke runs, six complete 200-epoch CPU runs and inference from saved checkpoints. Checks confirmed fixed split metadata and matching test-node IDs, agreement between saved training/test results and prediction metrics, consistent prediction counts and confidence decisions, and identical sampled IDs for the embedding figures. Error analysis examines actual predictions and graph neighbours. Proposed failure causes are identified as hypotheses where no ablation was performed. I am responsible for reviewing and understanding the submitted code, calculations and claims.

## References

1. Gayanku, [Facebook Page-Page dataset repository](https://github.com/gayanku/facebook_page-page_dataset), modified 128-feature archive; identity and class order verified for this experiment.
2. T. N. Kipf and M. Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR 2017.
3. PyTorch Geometric, [`GCNConv` documentation](https://pytorch-geometric.readthedocs.io/en/2.8.1/generated/torch_geometric.nn.conv.GCNConv.html). The implementation uses this library operator rather than reimplementing its message-passing primitive.
4. Original node labels, [`musae_facebook_target.csv`](https://github.com/gayanku/facebook_page-page_dataset/blob/main/facebook_large/musae_facebook_target.csv), used to check numeric-label names by node ID.
5. COMP3710/COMP7710 teaching team, *Pattern Recognition Report: Version 2.5 Final Assessment Specification*, course handout, classification guidelines and Sections 3-6.
