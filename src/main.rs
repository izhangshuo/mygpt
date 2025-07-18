#![recursion_limit = "256"]

use burn::{
    backend::{Autodiff, Wgpu, wgpu::WgpuDevice},
    config::Config,
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
};

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

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let artifact_dir = "artifact";

    let text = include_str!("input.txt");
    println!("text chars len: {}", text.chars().count());

    let tokenizer = Tokenizer::from_text(text);
    println!("vocab size: {}", tokenizer.vocab_size());
    println!("{}", tokenizer.decode(&tokenizer.encode("Hello world!")));

    let model =
        if let Some(config) = TrainingConfig::load(format!("{artifact_dir}/config.json")).ok() {
            let record = CompactRecorder::new()
                .load(format!("{artifact_dir}/model").into(), &device)
                .expect("Trained model should exist; run train first");

            config.model.init::<MyBackend>(&device).load_record(record)
        } else {
            train::<MyAutodiffBackend>(
                artifact_dir,
                tokenizer.encode(&text[0..10000]),
                TrainingConfig::new(
                    BigramModelConfig::new(tokenizer.vocab_size()),
                    AdamConfig::new(),
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

    let new_tokens = generate(&model, tokenizer.encode("A"), 100, &device);
    println!("{}", tokenizer.decode(&new_tokens));
}
