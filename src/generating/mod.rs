use async_stream::stream;
use burn::{
    prelude::Backend,
    tensor::{Int, Tensor, TensorData, activation::softmax},
};
use futures_core::Stream;

use crate::{MAX_BLOCK_SIZE, modeling::BigramModel};

pub fn generate<B: Backend>(
    model: &BigramModel<B>,
    prompt: Vec<i32>,
    max_new_tokens: usize,
    device: &B::Device,
) -> impl Stream<Item = i32> {
    stream! {
        use rand::distr::Distribution;
        // [0] ->[0,  1]   ->[0,  1,  2]
        // [0']->[0', 1']->[0', 1' ,2']

        let mut tokens = prompt;
        let mut rng = rand::rng();

        for _ in tokens.len()..tokens.len() + max_new_tokens {
            let start = tokens.len().saturating_sub(MAX_BLOCK_SIZE);
            let idx = tokens[start..].to_vec();
            let block_size = idx.len();

            let x = TensorData::new(idx, [1, block_size]);
            let x = Tensor::<B, 2, Int>::from_data(x, device);

            let logits = model.forward(x);
            let logits = logits.slice([0..1, block_size - 1..block_size]);

            let props = softmax(logits, 2);
            let props = props.to_data().to_vec::<f32>().unwrap();

            let distr = rand::distr::weighted::WeightedIndex::new(&props).unwrap();
            let next_token = distr.sample(&mut rng) as i32;

            yield next_token;

            tokens.push(next_token);
        }

    }
}
