use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, loss::CrossEntropyLoss},
    prelude::Backend,
    tensor::{Int, Tensor, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::batching::TokenPairBatch;

#[derive(Debug, Config)]
pub struct BigramModelConfig {
    vocab_size: usize,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        let token_embedding_table =
            EmbeddingConfig::new(self.vocab_size, self.vocab_size).init(device);

        BigramModel {
            token_embedding_table,
        }
    }
}

#[derive(Debug, Module)]
pub struct BigramModel<B: Backend> {
    token_embedding_table: Embedding<B>,
}

impl<B: Backend> BigramModel<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embedding_table.forward(idx)
    }

    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(inputs);

        let [b, t, c] = logits.dims(); // [4 * 8, 66]
        let logits = logits.reshape([b * t, c]);
        let targets = targets.reshape([b * t]);

        let loss = CrossEntropyLoss::new(None, &targets.device()) //
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, logits, targets)
    }

    pub fn loss(
        &self,
        idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let logits = self.forward(idx);

        let [b, t, c] = logits.dims(); // [4 * 8, 66]
        let logits = logits.reshape([b * t, c]);
        let targets = targets.reshape([b * t]);

        CrossEntropyLoss::new(None, device).forward(logits, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<TokenPairBatch<B>, ClassificationOutput<B>> for BigramModel<B> {
    fn step(&self, batch: TokenPairBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.inputs, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TokenPairBatch<B>, ClassificationOutput<B>> for BigramModel<B> {
    fn step(&self, batch: TokenPairBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.inputs, batch.targets)
    }
}
