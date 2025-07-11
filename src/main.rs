use std::collections::{BTreeSet, HashMap};

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    module::{AutodiffModule, Module},
    nn::{Embedding, EmbeddingConfig, loss::CrossEntropyLoss},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::Backend,
    tensor::{
        Int, Tensor, TensorData, activation::softmax, backend::AutodiffBackend, cast::ToElement,
    },
};
use rand::distr::Distribution;

pub struct Tokenizer {
    stoi: HashMap<char, i32>,
    itos: HashMap<i32, char>,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        let chars = text.chars().collect::<BTreeSet<char>>();

        let stoi = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (*ch, i as i32))
            .collect::<HashMap<char, i32>>();

        let itos = chars
            .iter()
            .enumerate()
            .map(|(i, ch)| (i as i32, *ch))
            .collect::<HashMap<i32, char>>();

        Self { stoi, itos }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        text.chars()
            .map(|ch| *self.stoi.get(&ch).unwrap())
            .collect()
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        tokens.iter().map(|i| *self.itos.get(&i).unwrap()).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}

pub enum BatchType {
    Train,
    Test,
}

pub struct Batcher {
    train_data: Vec<i32>,
    test_data: Vec<i32>,
}

impl Batcher {
    pub fn from_tokens(tokens: Vec<i32>) -> Self {
        let mut train_data = tokens;

        let split_at = (train_data.len() as f64 * 0.9) as usize;
        let test_data = train_data.split_off(split_at);

        println!("{}", train_data.len());
        println!("{}", test_data.len());

        Self {
            train_data,
            test_data,
        }
    }

    pub fn batch<B: Backend>(
        &self,
        typ: BatchType,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let data = match typ {
            BatchType::Train => &self.train_data,
            BatchType::Test => &self.test_data,
        };

        let mut rng = rand::rng();
        let ix = rand::seq::index::sample(
            &mut rng, //
            data.len() - BLOCK_SIZE - 1,
            BATCH_SIZE,
        );

        let x = ix
            .iter()
            .map(|i| data[i..i + BLOCK_SIZE].to_vec())
            .flatten()
            .collect();

        let y = ix
            .iter()
            .map(|i| data[i + 1..i + 1 + BLOCK_SIZE].to_vec())
            .flatten()
            .collect();

        let xd = TensorData::new(x, [BATCH_SIZE, BLOCK_SIZE]);
        let yd = TensorData::new(y, [BATCH_SIZE, BLOCK_SIZE]);

        let xt = Tensor::from_data(xd, device);
        let yt = Tensor::from_data(yd, device);

        (xt, yt)
    }
}

#[derive(Debug, Module)]
pub struct BigramModel<B: Backend> {
    token_embedding_table: Embedding<B>,
}

impl<B: Backend> BigramModel<B> {
    pub fn new(vocab_size: usize, device: &B::Device) -> Self {
        let token_embedding_table = EmbeddingConfig::new(vocab_size, vocab_size).init(device);

        Self {
            token_embedding_table,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.token_embedding_table.forward(idx)
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

pub fn generate<B: Backend>(
    model: &BigramModel<B>,
    prompt: Vec<i32>,
    max_new_tokens: usize,
    device: &B::Device,
) -> Vec<i32> {
    // [0] ->[0,  1]   ->[0,  1,  2]
    // [0']->[0', 1']->[0', 1' ,2']

    let mut tokens = prompt;
    let mut rng = rand::rng();

    for i in tokens.len()..tokens.len() + max_new_tokens {
        let x = TensorData::new(tokens.clone(), [1, i]);
        let x = Tensor::<B, 2, Int>::from_data(x, device);

        let logits = model.forward(x);
        let logits = logits.slice([0..1, i - 1..i]);

        let props = softmax(logits, 2);
        let props = props.to_data().to_vec::<f32>().unwrap();

        let distr = rand::distr::weighted::WeightedIndex::new(&props).unwrap();
        let next_token = distr.sample(&mut rng) as i32;

        tokens.push(next_token);
    }

    tokens
}

pub fn train<B: AutodiffBackend>(
    vocab_size: usize,
    batcher: &Batcher,
    device: &B::Device,
) -> BigramModel<B> {
    let mut model = BigramModel::<B>::new(vocab_size, device);
    let mut optimizer = AdamWConfig::new().init::<B, BigramModel<B>>();

    for i in 0..10000 {
        let (x, y) = batcher.batch::<B>(BatchType::Train, device);

        let loss = model.loss(x, y, device);

        if i % 1000 == 0 {
            let last_loss = loss.clone().into_scalar().to_f32();
            println!("loss: {last_loss}");
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);

        model = optimizer.step(1e-3, model, grads);
    }

    model
}

const BATCH_SIZE: usize = 4;
const BLOCK_SIZE: usize = 8;

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();

    let text = include_str!("input.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("{}", tokenizer.decode(&tokenizer.encode("Hello world!")));

    let batcher = Batcher::from_tokens(tokenizer.encode(text));
    let (x, y) = batcher.batch::<MyBackend>(BatchType::Train, &device);

    println!("{x}");
    println!("{y}");

    let model = BigramModel::<MyBackend>::new(tokenizer.vocab_size(), &device);
    let logits = model.forward(x.clone());
    println!("{logits}");

    let loss = model.loss(x, y, &device);
    println!("{loss}");

    let new_tokens = generate(&model, tokenizer.encode("A"), 100, &device);
    println!("{}", tokenizer.decode(&new_tokens));

    let trained_model = train::<MyAutodiffBackend>(
        tokenizer.vocab_size(), //
        &batcher,
        &device,
    )
    .valid();

    let new_tokens = generate(&trained_model, tokenizer.encode("A"), 100, &device);
    println!("{}", tokenizer.decode(&new_tokens));
}
