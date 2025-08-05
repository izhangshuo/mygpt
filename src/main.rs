#![recursion_limit = "256"]

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    config::Config,
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
};

use futures_util::{pin_mut, stream::StreamExt};

use crate::{
    generating::generate,
    modeling::BigramModelConfig,
    tokenizing::Tokenizer,
    training::{TrainingConfig, train},
};

pub mod batching;
pub mod dataset;
pub mod generating;
pub mod modeling;
pub mod tokenizing;
pub mod training;

pub const MAX_BLOCK_SIZE: usize = 256;
pub const N_EMBD: usize = 384;
pub const N_HEADS: usize = 6;
pub const N_LAYER: usize = 6;
pub const DROPOUT: f64 = 0.2;
pub const NUM_EPOCHS: usize = 1;
pub const BATCH_SIZE: usize = 64;
pub const NUM_WORKERS: usize = 20;
pub const LEARNING_RATE: f64 = 3.0e-4;

#[tokio::main]
async fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "artifact";

    let text = include_str!("4in1.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());

    let model =
        if let Some(config) = TrainingConfig::load(format!("{artifact_dir}/config.json")).ok() {
            let record = CompactRecorder::new()
                .load(format!("{artifact_dir}/model").into(), &device)
                .expect("Trained model should exist; run train first");

            config.model.init::<MyBackend>(&device).load_record(record)
        } else {
            train::<MyAutodiffBackend>(
                artifact_dir,
                tokenizer.encode(&text),
                TrainingConfig::new(
                    BigramModelConfig::new(
                        tokenizer.vocab_size(), //
                        MAX_BLOCK_SIZE,
                        N_EMBD,
                        N_HEADS,
                        N_LAYER,
                        DROPOUT,
                    ),
                    AdamConfig::new(),
                    NUM_EPOCHS,
                    BATCH_SIZE,
                    NUM_WORKERS,
                    LEARNING_RATE,
                ),
                device.clone(),
            );

            let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
                .expect("Config should exist for the model; run train first");
            let record = CompactRecorder::new()
                .load(format!("{artifact_dir}/model").into(), &device)
                .expect("Trained model should exist; run train first");

            config.model.init::<MyBackend>(&device).load_record(record)
        };

    let stream = generate(&model, tokenizer.encode("大师兄"), 10000, &device);
    pin_mut!(stream); // needed for iteration

    while let Some(token) = stream.next().await {
        print!("{}", tokenizer.decode(&[token]));
    }
}
