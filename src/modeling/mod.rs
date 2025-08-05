use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig, Relu, loss::CrossEntropyLoss,
    },
    prelude::Backend,
    tensor::{Int, Tensor, activation::softmax, backend::AutodiffBackend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::batching::TokenPairBatch;

#[derive(Debug, Config)]
pub struct HeadConfig {
    n_embd: usize,
    head_size: usize,
    dropout: f64,
}

impl HeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Head<B> {
        let query = LinearConfig::new(
            self.n_embd, //
            self.head_size,
        )
        .init(device);

        let key = LinearConfig::new(
            self.n_embd, //
            self.head_size,
        )
        .init(device);

        let value = LinearConfig::new(
            self.n_embd, //
            self.head_size,
        )
        .init(device);

        let dropout = DropoutConfig::new(self.dropout).init();

        Head {
            query,
            key,
            value,
            dropout,
        }
    }
}

#[derive(Debug, Module)]
pub struct Head<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> Head<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let device = x.device();

        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x.clone());

        let [_, _, c] = k.dims();
        let wei = q.matmul(k.transpose()).div_scalar((c as f32).sqrt());
        let mask = Tensor::tril_mask(
            [b, t, t], //
            0,
            &device,
        );
        let wei = wei.mask_fill(mask, -f32::INFINITY);
        let wei = softmax(wei, 2);
        let wei = self.dropout.forward(wei);

        let x = wei.matmul(v);

        x
    }
}

#[derive(Debug, Config)]
pub struct MultiHeadConfig {
    n_embd: usize,
    n_heads: usize,
    dropout: f64,
}

impl MultiHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHead<B> {
        let heads = (0..self.n_heads)
            .map(|_| {
                HeadConfig::new(
                    self.n_embd, //
                    self.n_embd / self.n_heads,
                    self.dropout,
                )
                .init(device)
            })
            .collect();

        let proj = LinearConfig::new(self.n_embd, self.n_embd).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        MultiHead {
            heads,
            proj,
            dropout,
        }
    }
}

#[derive(Debug, Module)]
pub struct MultiHead<B: Backend> {
    heads: Vec<Head<B>>,
    proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> MultiHead<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let xs = self.heads.iter().map(|v| v.forward(x.clone())).collect();
        let x = Tensor::cat(xs, 2).to_device(&x.device());
        let x = self.proj.forward(x);
        let x = self.dropout.forward(x);

        x
    }
}

#[derive(Debug, Config)]
pub struct FeedForwardConfig {
    n_embd: usize,
    dropout: f64,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let linear = LinearConfig::new(self.n_embd, self.n_embd * 4).init(device);
        let relu = Relu::new();
        let proj = LinearConfig::new(self.n_embd * 4, self.n_embd).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        FeedForward {
            linear,
            relu,
            proj,
            dropout,
        }
    }
}

#[derive(Debug, Module)]
pub struct FeedForward<B: Backend> {
    linear: Linear<B>,
    relu: Relu,
    proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear.forward(x);
        let x = self.relu.forward(x);
        let x = self.proj.forward(x);
        let x = self.dropout.forward(x);

        x
    }
}

#[derive(Debug, Config)]
pub struct BlockConfig {
    n_embd: usize,
    n_heads: usize,
    dropout: f64,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        let sa_heads = MultiHeadConfig::new(
            self.n_embd, //
            self.n_heads,
            self.dropout,
        )
        .init(device);

        let ffwd = FeedForwardConfig::new(
            self.n_embd, //
            self.dropout,
        )
        .init(device);

        let ln1 = LayerNormConfig::new(self.n_embd).init(device);
        let ln2 = LayerNormConfig::new(self.n_embd).init(device);

        Block {
            sa_heads,
            ffwd,
            ln1,
            ln2,
        }
    }
}

#[derive(Debug, Module)]
pub struct Block<B: Backend> {
    sa_heads: MultiHead<B>,
    ffwd: FeedForward<B>,
    ln1: LayerNorm<B>,
    ln2: LayerNorm<B>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.sa_heads.forward(self.ln1.forward(x.clone()));
        let x = x.clone() + self.ffwd.forward(self.ln2.forward(x.clone()));

        x
    }
}

#[derive(Debug, Config)]
pub struct BigramModelConfig {
    vocab_size: usize,
    pub block_size: usize,
    n_embd: usize,
    n_heads: usize,
    n_layer: usize,
    dropout: f64,
}

impl BigramModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BigramModel<B> {
        let token_embedding_table = EmbeddingConfig::new(
            self.vocab_size, //
            self.n_embd,
        )
        .init(device);

        let position_embedding_table = EmbeddingConfig::new(
            self.block_size, //
            self.n_embd,
        )
        .init(device);

        let blocks = (0..self.n_layer)
            .map(|_| {
                BlockConfig::new(
                    self.n_embd, //
                    self.n_heads,
                    self.dropout,
                )
                .init(device)
            })
            .collect();

        let lm_head = LinearConfig::new(
            self.n_embd, //
            self.vocab_size,
        )
        .init(device);

        BigramModel {
            token_embedding_table,
            position_embedding_table,
            blocks,
            lm_head,
        }
    }
}

#[derive(Debug, Module)]
pub struct BigramModel<B: Backend> {
    token_embedding_table: Embedding<B>,
    position_embedding_table: Embedding<B>,
    blocks: Vec<Block<B>>,
    lm_head: Linear<B>,
}

impl<B: Backend> BigramModel<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = idx.dims();
        let device = idx.device();

        let tok_emb = self.token_embedding_table.forward(idx);

        let pos_idx = Tensor::arange(0..t as i64, &device)
            .reshape([1, t])
            .repeat_dim(0, b);
        let pos_emb = self.position_embedding_table.forward(pos_idx);

        let x = tok_emb + pos_emb;
        let x = self.blocks.iter().fold(x, |acc, item| item.forward(acc));
        let x = self.lm_head.forward(x);

        x
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

#[cfg(test)]
mod tests {
    use core::f32;
    use std::ops::Add;

    use burn::{
        backend::{Wgpu, wgpu::WgpuDevice},
        tensor::{Bool, Tensor, activation::softmax},
    };

    #[test]
    pub fn test_mask_ok() {
        type MyBackend = Wgpu;

        let device = WgpuDevice::default();

        let ones = Tensor::<MyBackend, 2>::ones([4, 4], &device);
        let mask = Tensor::<MyBackend, 2, Bool>::tril_mask([4, 4], 0, &device);
        let ones = ones.mask_fill(mask, -f32::INFINITY);
        let ones = softmax(ones, 1);

        println!("ones: {ones}");
        // println!("ones: {mask}");
    }

    #[test]
    pub fn test_sequential_ok() {
        let res = (0..4).fold(0, |acc, item| item.add(acc));

        // 1 - item=0, acc=0, 0.add(0) = 0
        // 2 - item=1, acc=0, 1.add(0) = 1
        // 3 - item=2, acc=1, 2.add(1) = 3
        // 4 - item=3, acc=3, 3.add(3) = 6

        // fn4(fn3(fn2(fn1(x))))
        println!("res: {res}");
    }
}
